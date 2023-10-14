from transformers import RoFormerForCausalLM, RoFormerModel, RoFormerConfig, GenerationConfig
import torch
from torch import nn
from text.symbols import *

from cluster import get_cluster_model, get_cluster_result, get_cluster_center_result, get_center

from torch.nn.utils.rnn import pad_sequence, pack_sequence

from copy import deepcopy

def get_model(mode = "phone", semantic_kmeans_num = 10000, codebook_path = "pretrain/semantic_codebook.pt", n_spk = 1, **kwargs):
    config = RoFormerConfig(
            hidden_size=kwargs["model"]["hidden_size"],
            num_attention_heads=kwargs["model"]["num_attention_heads"],
            num_hidden_layers=kwargs["model"]["num_hidden_layers"],
            intermediate_size=kwargs["model"]["intermediate_size"],
            hidden_act=kwargs["model"]["hidden_act"],
            hidden_dropout_prob=kwargs["model"]["hidden_dropout_prob"],
            attention_probs_dropout_prob=kwargs["model"]["attention_probs_dropout_prob"],
            initializer_range=kwargs["model"]["initializer_range"],
            layer_norm_eps=float(kwargs["model"]["layer_norm_eps"]),
            max_position_embeddings = kwargs["model"]["max_position_embeddings"]
        )
    
    model = Roformer(
        config = config,
        mode = mode,
        semantic_kmeans_num = semantic_kmeans_num,
        codebook_path = codebook_path,
        n_spk = n_spk
    )

    return model
    
    

class Roformer(nn.Module):
    def __init__(
        self,
        config: RoFormerConfig,
        mode = "phone",
        semantic_kmeans_num = 10000,
        codebook_path = "pretrain/semantic_codebook.pt",
        n_spk = 1,
        **kwargs
        ):
        super().__init__()
        self.mode = mode
        self.config = config
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
        config.vocab_size = token_size
        config.type_vocab_size = self.num_tones + 1
        config.pad_token_id = self.PAD
        config.bos_token_id = self.BOS
        config.eos_token_id = self.EOS
        self.text_encoder = RoFormerModel(config)

        config = deepcopy(config)
        
        config.bos_token_id = semantic_kmeans_num
        config.eos_token_id = semantic_kmeans_num + 1
        config.pad_token_id = semantic_kmeans_num + 2
        config.vocab_size = semantic_kmeans_num + 3

        self.semantic_bos_token_id = semantic_kmeans_num
        self.semantic_eos_token_id = semantic_kmeans_num + 1
        self.semantic_pad_token_id = semantic_kmeans_num + 2

        config.type_vocab_size = 1
        config.is_decoder = True
        config.add_cross_attention = True
        self.semantic_decoder = RoFormerForCausalLM(config)
        
        self.quantizer = get_cluster_model(codebook_path)

        if self.semantic_decoder.roformer.embeddings.word_embeddings.weight.data.shape[1] == self.quantizer.cluster_centers_.shape[1]:
            self.semantic_decoder.roformer.embeddings.word_embeddings.weight.data[:semantic_kmeans_num] = torch.from_numpy(self.quantizer.cluster_centers_.copy())

        if n_spk > 1 and n_spk is not None:
            self.spk_emb = nn.Embedding(n_spk + 1, config.hidden_size)
        else:
            self.spk_emb = None

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
                **kwargs
                 ):
        phone_tone_emb = self.text_encoder.embeddings(phone,tone)
        
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
            bos_token_id = self.semantic_bos_token_id
        )

        outputs = self.semantic_decoder.generate(
            encoder_hidden_states = encoder_hidden_states,
            encoder_attention_mask = attention_mask,
            attention_mask = None,
            use_cache = use_cache,
            generation_config=generation_config
        )

        return outputs



if __name__ == '__main__':
    a = RoFormerConfig(
         hidden_size=768,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_hidden_groups=1,
            intermediate_size=512,
    )
    b = Roformer(config=a)
    phone = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
    tone = torch.LongTensor([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    semantic = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
    labels = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
    outputs = b(phone=phone, tone=tone, semantic=semantic,labels=labels)
    print(outputs)
    generate = b.generate(phone=phone, tone=tone, attention_mask=None)
    print(generate)

