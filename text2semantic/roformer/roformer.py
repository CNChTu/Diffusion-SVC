from transformers import RoFormerForCausalLM, RoFormerModel, RoFormerConfig, GenerationConfig
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.models.roformer.modeling_roformer import RoFormerSelfAttention
import torch
from torch import nn
from text.symbols import *
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from cluster import get_cluster_model


from copy import deepcopy

def get_model(mode = "phone", semantic_kmeans_num = 10000, codebook_path = "pretrain/semantic_codebook.pt", n_spk = 1, **kwargs):
    encoder_config = RoFormerConfig(
            hidden_size=kwargs["model"]["encoder"]["hidden_size"],
            num_attention_heads=kwargs["model"]["encoder"]["num_attention_heads"],
            num_hidden_layers=kwargs["model"]["encoder"]["num_hidden_layers"],
            intermediate_size=kwargs["model"]["encoder"]["intermediate_size"],
            hidden_act=kwargs["model"]["encoder"]["hidden_act"],
            hidden_dropout_prob=kwargs["model"]["encoder"]["hidden_dropout_prob"],
            attention_probs_dropout_prob=kwargs["model"]["encoder"]["attention_probs_dropout_prob"],
            initializer_range=kwargs["model"]["encoder"]["initializer_range"],
            layer_norm_eps=float(kwargs["model"]["encoder"]["layer_norm_eps"]),
            max_position_embeddings = kwargs["model"]["encoder"]["max_position_embeddings"],
            is_decoder = False
        )
    
    decoder_config = RoFormerConfig(
            hidden_size=kwargs["model"]["decoder"]["hidden_size"],
            num_attention_heads=kwargs["model"]["decoder"]["num_attention_heads"],
            num_hidden_layers=kwargs["model"]["decoder"]["num_hidden_layers"],
            intermediate_size=kwargs["model"]["decoder"]["intermediate_size"],
            hidden_act=kwargs["model"]["decoder"]["hidden_act"],
            hidden_dropout_prob=kwargs["model"]["decoder"]["hidden_dropout_prob"],
            attention_probs_dropout_prob=kwargs["model"]["decoder"]["attention_probs_dropout_prob"],
            initializer_range=kwargs["model"]["decoder"]["initializer_range"],
            layer_norm_eps=float(kwargs["model"]["decoder"]["layer_norm_eps"]),
            max_position_embeddings = kwargs["model"]["decoder"]["max_position_embeddings"],
            is_decoder = True
    )

    model = Roformer(
        encoder_config = encoder_config,
        decoder_config = decoder_config,
        mode = mode,
        semantic_kmeans_num = semantic_kmeans_num,
        codebook_path = codebook_path,
        n_spk = n_spk,
        use_flash_attn = kwargs["use_flash_attn"]
    )

    return model
    
class EndGateLogitsProcessor(LogitsProcessor):
    def __init__(self, end_gate_threshold: float, eos_token_id: int):
        
        self.end_gate_threshold = end_gate_threshold
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        gate = torch.softmax(scores,dim=-1)[:, self.eos_token_id] > self.end_gate_threshold
        scores[gate] = float("inf")
        return scores



class Roformer(nn.Module):
    def __init__(
        self,
        encoder_config: RoFormerConfig,
        decoder_config: RoFormerConfig,
        mode = "phone",
        semantic_kmeans_num = 10000,
        codebook_path = "pretrain/semantic_codebook.pt",
        n_spk = 1,
        use_flash_attn = False,
        **kwargs
        ):
        super().__init__()
        self.mode = mode
        self.n_spk = n_spk
        if "phone" in self.mode:
            token_size = len(symbols)
            # token_size += semantic_kmeans_num + num_tones
            self.BOS = token_size
            self.EOS = token_size + 1
            self.PAD = token_size + 2
            self.num_tones = num_tones
            token_size += 3
            # self.tone_emb = nn.Embedding(num_tones, config.hidden_size)
            # self.phone_emb = nn.Embedding(token_size + 2, config.hidden_size)
        if "text" in self.mode:
            from transformers import BertTokenizer
            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir="./pretrain")
            token_size = bert_tokenizer.vocab_size
            self.BOS = bert_tokenizer.cls_token_id
            self.EOS = token_size.sep_token_id
            self.PAD = token_size.pad_token_id
        encoder_config.vocab_size = token_size
        encoder_config.type_vocab_size = self.num_tones + 1
        encoder_config.pad_token_id = self.PAD
        encoder_config.bos_token_id = self.BOS
        encoder_config.eos_token_id = self.EOS
        self.text_encoder = RoFormerModel(encoder_config)

        decoder_config.bos_token_id = semantic_kmeans_num
        decoder_config.eos_token_id = semantic_kmeans_num + 1
        decoder_config.pad_token_id = semantic_kmeans_num + 2
        decoder_config.vocab_size = semantic_kmeans_num + 3

        self.semantic_bos_token_id = semantic_kmeans_num
        self.semantic_eos_token_id = semantic_kmeans_num + 1
        self.semantic_pad_token_id = semantic_kmeans_num + 2

        decoder_config.type_vocab_size = 1
        decoder_config.add_cross_attention = True
        self.semantic_decoder = RoFormerForCausalLM(decoder_config)
        self.semantic_decoder.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        
        self.quantizer = get_cluster_model(codebook_path)

        if self.semantic_decoder.roformer.embeddings.word_embeddings.weight.data.shape[1] == self.quantizer.cluster_centers_.shape[1]:
            self.semantic_decoder.roformer.embeddings.word_embeddings.weight.data[:semantic_kmeans_num] = torch.from_numpy(self.quantizer.cluster_centers_.copy())

        if n_spk > 1 and n_spk is not None:
            self.spk_emb = nn.Embedding(n_spk + 1, encoder_config.hidden_size)
        else:
            self.spk_emb = None

        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.semantic_decoder.roformer.get_extended_attention_mask = self.get_flash_attn_extended_attention_mask
            for i in self.text_encoder.encoder.layer:
                i.attention.self = RoFormerFlashAttention(encoder_config)
            for i in self.semantic_decoder.roformer.encoder.layer:
                i.attention.self = RoFormerFlashAttention(decoder_config)
                i.crossattention.self = RoFormerFlashAttention(decoder_config)
    
    def get_flash_attn_extended_attention_mask(self, attention_mask, input_shape, dtype = None):
        if dtype is None:
            dtype = self.text_encoder.embeddings.word_embeddings.weight.dtype
        
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask
    

    def forward(
        self,
        phone,
        tone,
        semantic,
        attention_mask=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        spk_id=None,
        **kwargs
    ):
        if self.spk_emb is not None and spk_id is not None:
            spk_emb = self.spk_emb(spk_id)
        else:
            spk_emb = 0

        phone_tone_emb = self.text_encoder.embeddings(phone,tone) + spk_emb
        
        encoder_hidden_states = self.text_encoder(
            inputs_embeds = phone_tone_emb,
            attention_mask = encoder_attention_mask,
            use_cache = use_cache
        ).last_hidden_state
        
        outputs = self.semantic_decoder(
            semantic,
            encoder_hidden_states = encoder_hidden_states,
            encoder_attention_mask = encoder_attention_mask,
            attention_mask = attention_mask,
            labels = labels,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            use_cache = use_cache
        )
        return outputs
    
    @torch.no_grad()
    def generate(self,
                 phone,
                 tone,
                 attention_mask=None,
                 use_cache=None,
                 max_length=1024,
                 do_sample=True,
                 temperature=1.0,
                 top_k=5,
                 top_p=1.0,
                 repetition_penalty=1.0,
                 num_beams=1,
                 no_repeat_ngram_size = 0,
                 early_stopping = True,
                 spk_id = None,
                 end_gate_threshold = None,
                 **kwargs
                 ):
        if self.use_flash_attn:
            use_cache = False
        logits_processor = LogitsProcessorList()
        if end_gate_threshold is not None:
            logits_processor.append(EndGateLogitsProcessor(end_gate_threshold = end_gate_threshold, eos_token_id = self.semantic_eos_token_id))

        if len(logits_processor) == 0:
            logits_processor = None

        if self.spk_emb is not None and spk_id is not None:
            spk_emb = self.spk_emb(spk_id)
        else:
            spk_emb = 0
        
        phone_tone_emb = self.text_encoder.embeddings(phone,tone) + spk_emb
        
        encoder_hidden_states = self.text_encoder(
            inputs_embeds = phone_tone_emb,
            attention_mask = attention_mask,
            use_cache = use_cache
        )[0]

        if num_beams == 1:
            early_stopping = False

        generation_config = GenerationConfig(
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            no_repeat_ngram_size = no_repeat_ngram_size,
            early_stopping = early_stopping,
            bos_token_id = self.semantic_bos_token_id,
            eos_token_id = self.semantic_eos_token_id,
            pad_token_id = self.semantic_pad_token_id
        )

        outputs = self.semantic_decoder.generate(
            encoder_hidden_states = encoder_hidden_states,
            encoder_attention_mask = attention_mask,
            attention_mask = None,
            use_cache = use_cache,
            generation_config=generation_config,
            logits_processor = logits_processor
        )

        return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values, **model_kwargs}


class RoFormerFlashAttention(RoFormerSelfAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        from flash_attn import flash_attn_varlen_func
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            # attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            # attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            if sinusoidal_pos is not None:
                if self.rotary_value:
                    query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer, value_layer
                    )
                else:
                    query_layer, key_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer
                    )
            if past_key_value is not None:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        source_dtype = query_layer.dtype
        
        if encoder_attention_mask is None:
            encoder_attention_mask = attention_mask

        max_q_len = query_layer.size(2)
        max_k_len = key_layer.size(2)

        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)
        _,_,n_haed,head_dim = query_layer.shape
        query_layer = query_layer.view(-1, n_haed, head_dim).contiguous()
        key_layer = key_layer.view(-1, n_haed, head_dim).contiguous()
        value_layer = value_layer.view(-1, n_haed, head_dim).contiguous()
        
        q_len = torch.sum((attention_mask==0), dim=-1)[:,0,0].int()
        k_len = torch.sum((encoder_attention_mask==0), dim=-1)[:,0,0].int()
        
        attention_index = (attention_mask==0).view(-1)
        encoder_attention_index = (encoder_attention_mask==0).view(-1)
        query_layer = query_layer[attention_index]
        key_layer = key_layer[encoder_attention_index]
        value_layer = value_layer[encoder_attention_index]
        
        query_layer = query_layer.to(torch.bfloat16)
        key_layer = key_layer.to(torch.bfloat16)
        value_layer = value_layer.to(torch.bfloat16)
        
        context_layer, attention_probs, S_dmask = flash_attn_varlen_func(query_layer, key_layer, value_layer,q_len,k_len,max_q_len, max_k_len, self.dropout.p, causal=self.is_decoder, return_attn_probs=True)
        assert not torch.isnan(context_layer).any(), "context_layer contains NaN"
        assert not torch.isnan(attention_probs).any(), "attention_probs contains NaN"
        context_layer = context_layer.to(source_dtype)
        attention_probs = attention_probs.to(source_dtype)
         
        context_layer = [context_layer[start:start+length,...] for start, length in zip([0]+q_len[:-1].tolist(), q_len.tolist())]
        
        context_layer = pad_sequence(context_layer, batch_first=True, padding_value=0)

        if context_layer.shape[1] < max_q_len:
            context_layer = F.pad(context_layer, (0,0,0,0,0,max_q_len-context_layer.shape[1]), value=0)
        
        context_layer = context_layer.contiguous()
        # Mask heads if we want to

        if head_mask is not None:
            context_layer *= head_mask

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

if __name__ == '__main__':
    a = RoFormerConfig(
         hidden_size=768,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_hidden_groups=1,
            intermediate_size=512,
            is_decoder = False
    )
    a2 = RoFormerConfig(
         hidden_size=768,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_hidden_groups=1,
            intermediate_size=512,
            is_decoder = True,
            add_cross_attention = True
    )
    b = Roformer(a,a2, semantic_kmeans_num=2048, use_flash_attn=True).cuda()
    phone = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]).cuda()
    tone = torch.LongTensor([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]).cuda()
    semantic = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]).cuda()
    labels = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]).cuda()
    attention_mask = torch.LongTensor([[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]]).cuda()
    outputs = b(phone=phone, tone=tone, semantic=semantic,labels=labels,attention_mask=attention_mask,encoder_attention_mask=attention_mask)
    print(outputs)
    generate = b.generate(phone=phone, tone=tone,end_gate_threshold=0.9)
    print(generate)

