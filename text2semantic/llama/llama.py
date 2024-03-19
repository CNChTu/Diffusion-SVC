from transformers import LlamaForCausalLM, LlamaConfig, GenerationConfig
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
import torch
from torch import nn
from text.symbols import *

from cluster import get_cluster_model, get_cluster_result, get_cluster_center_result, get_center

from torch.nn.utils.rnn import pad_sequence, pack_sequence

def get_model(mode = "phone", semantic_kmeans_num = 10000, codebook_path = "pretrain/semantic_codebook.pt", n_spk = 1, **kwargs):
    decoder_config = LlamaConfig(
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

    model = Llama(
        config = decoder_config,
        mode = mode,
        semantic_kmeans_num = semantic_kmeans_num,
        codebook_path = codebook_path,
        n_spk = n_spk,
        use_flash_attn = kwargs["use_flash_attn"],
        gradient_checkpointing = kwargs["model"]["gradient_checkpointing"]
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


class Llama(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        mode = "phone",
        semantic_kmeans_num = 10000,
        n_spk = 1,
        codebook_path = "pretrain/semantic_codebook.pt",
        use_flash_attn = False,
        gradient_checkpointing = False
        ):
        super().__init__()
        self.mode = mode
        self.config = config
        token_shift = 0
        if "phone" in self.mode:
            token_size = len(symbols)
            # token_size += semantic_kmeans_num + num_tones
            self.BOS = token_size
            self.EOS = token_size + 1
            self.PAD = token_size + 2
            self.tone_token_shift = token_size + 3
            self.num_tones = num_tones
            self.TONE_BOS = self.tone_token_shift + self.num_tones
            self.TONE_EOS = self.tone_token_shift + self.num_tones + 1
            self.TONE_PAD = self.tone_token_shift + self.num_tones + 2
            token_size += self.num_tones + 3
            # self.tone_emb = nn.Embedding(num_tones, config.hidden_size)
            # self.phone_emb = nn.Embedding(token_size + 2, config.hidden_size)
        if "text" in self.mode:
            from transformers import BertTokenizer
            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir="./pretrain")
            token_size = bert_tokenizer.vocab_size
            self.BOS = bert_tokenizer.cls_token_id
            self.EOS = bert_tokenizer.sep_token_id
            self.PAD = bert_tokenizer.pad_token_id
            self.num_tones = 0
        token_shift = token_size
        self.semantic_token_shift = token_shift

        config.bos_token_id = self.semantic_token_shift + semantic_kmeans_num
        config.eos_token_id = self.semantic_token_shift + semantic_kmeans_num + 1
        config.pad_token_id = self.semantic_token_shift + semantic_kmeans_num + 2

        self.semantic_bos_token_id = config.bos_token_id
        self.semantic_eos_token_id = config.eos_token_id
        self.semantic_pad_token_id = config.pad_token_id

        token_size += semantic_kmeans_num + 3

        config.vocab_size = token_size
        # llama用flash attention 2.0
        if use_flash_attn:
            config._attn_implementation = "flash_attention_2"
        self.llama = LlamaForCausalLM(config=config)
        if gradient_checkpointing:
            self.llama.gradient_checkpointing_enable()
            
        try:
            self.quantizer = get_cluster_model(codebook_path)

            if self.llama.model.embed_tokens.weight.data.shape[1] == self.quantizer.cluster_centers_.shape[1]:
                self.llama.model.embed_tokens.weight.data[len(symbols) - 1:len(symbols) + semantic_kmeans_num - 1] = torch.from_numpy(self.quantizer.cluster_centers_.copy())
        except:
            pass

    def forward(
        self,
        phone=None,
        tone=None,
        semantic=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        if input_ids is None:
            B,T = phone.shape
            if self.mode == "phone":
                tone += self.tone_token_shift
                phone = torch.cat([torch.tensor([[self.BOS]]), phone, torch.tensor([[self.EOS]]), torch.tensor([[self.TONE_BOS]]), tone, torch.tensor([[self.TONE_EOS]])], dim=1)
            elif self.mode == "text":
                phone = phone
                
            semantic += self.semantic_token_shift
            semantic = torch.cat([torch.tensor([[self.config.bos_token_id]]), semantic, torch.tensor([[self.config.eos_token_id]])], dim=1)

            input_ids = torch.cat([phone, semantic], dim=1)
        
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
            

        return outputs
    
    
    @torch.no_grad()
    def generate(self,
                 phone=None,
                 tone=None,
                 input_ids=None,
                 attention_mask=None,
                 use_cache=True,
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
        
        logits_processor = LogitsProcessorList()
        if end_gate_threshold is not None:
            logits_processor.append(EndGateLogitsProcessor(end_gate_threshold = end_gate_threshold, eos_token_id = self.config.eos_token_id))

        if len(logits_processor) == 0:
            logits_processor = None
        
        if phone is not None:
            if self.mode == "phone":
                tone += self.tone_token_shift
                phone = torch.cat([torch.tensor([[self.BOS]],device=phone.device), phone, torch.tensor([[self.EOS]],device=phone.device), torch.tensor([[self.TONE_BOS]],device=phone.device), tone, torch.tensor([[self.TONE_EOS]],device=phone.device)], dim=1)
            elif self.mode == "text":
                phone = phone
            input_ids = torch.cat([phone, torch.tensor([[self.config.bos_token_id]],device=phone.device)], dim=1)
        
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
            bos_token_id = self.config.bos_token_id,
            eos_token_id = self.config.eos_token_id,
            pad_token_id = self.config.pad_token_id
        )

        outputs = self.llama.generate(
            inputs = input_ids,
            encoder_attention_mask = attention_mask,
            attention_mask = None,
            use_cache = use_cache,
            generation_config=generation_config,
            logits_processor = logits_processor
        )

        if (outputs >= self.config.bos_token_id).any():
            outputs[outputs >= self.config.bos_token_id] = self.config.bos_token_id - 1
        outputs = outputs[:, input_ids.shape[1]:] - self.semantic_token_shift

        return outputs
if __name__ == '__main__':
    a = LlamaConfig(
         hidden_size=768,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_hidden_groups=1,
            intermediate_size=512,
    )
    b = Llama(config=a,semantic_kmeans_num=2048)
    phone = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
    tone = torch.LongTensor([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    semantic = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
    labels = torch.cat([torch.tensor([[b.BOS]]), phone, torch.tensor([[b.EOS]]), torch.tensor([[b.TONE_BOS]]), tone, torch.tensor([[b.TONE_EOS]]), torch.tensor([[b.config.bos_token_id]]), semantic, torch.tensor([[b.config.eos_token_id]])], dim=1)
    outputs = b(phone=phone, tone=tone, semantic=semantic,labels=labels)
    print(outputs)
    generate = b.generate(phone=phone, tone=tone,end_gate_threshold=0.9)
    print(generate)


