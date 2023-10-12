from transformers import RoFormerForCausalLM, RoFormerModel, RoFormerConfig
import torch
from torch import nn
from text.symbols import *

from cluster import get_cluster_model, get_cluster_result, get_cluster_center_result, get_center

from torch.nn.utils.rnn import pad_sequence, pack_sequence


class Roformer(nn.Module):
    def __init__(
        self,
        config: RoFormerConfig,
        mode = "phone",
        semantic_kmeans_num = 10000,
        codebook_path = "pretrain/semantic_codebook.pt",
        ):
        super().__init__()
        self.mode = mode
        self.config = config
        if "phone" in self.mode:
            token_size = len(symbols)
            # token_size += semantic_kmeans_num + num_tones
            self.BOS = token_size
            self.EOS = token_size + 1
            self.PAD = token_size + 2
            token_size += 3
            # self.tone_emb = nn.Embedding(num_tones, config.hidden_size)
            # self.phone_emb = nn.Embedding(token_size + 2, config.hidden_size)
        config.vocab_size = token_size
        config.type_vocab_size = num_tones + 1
        config.pad_token_id = self.PAD
        config.bos_token_id = self.BOS
        config.eos_token_id = self.EOS
        self.text_encoder = RoFormerModel(config)

        config.bos_token_id = semantic_kmeans_num
        config.eos_token_id = semantic_kmeans_num + 1
        config.pad_token_id = semantic_kmeans_num + 2
        config.vocab_size = semantic_kmeans_num + 3

        config.type_vocab_size = 1
        config.is_decoder = True
        config.add_cross_attention = True
        self.semantic_decoder = RoFormerForCausalLM(config)
        
        self.quantizer = get_cluster_model(codebook_path)

        if self.semantic_decoder.roformer.embeddings.word_embeddings.weight.data.shape[1] == self.quantizer.cluster_centers_.shape[1]:
            self.semantic_decoder.roformer.embeddings.word_embeddings.weight.data[:semantic_kmeans_num] = torch.from_numpy(self.quantizer.cluster_centers_.copy())


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
        **kwargs
    ):
        B,T = phone.shape
        phone_tone_emb = self.text_encoder.embeddings(phone,tone)

        encoder_hidden_states = self.text_encoder(
            inputs_embeds = phone_tone_emb,
            attention_mask = encoder_attention_mask,
            use_cache = use_cache
        )[0]

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

