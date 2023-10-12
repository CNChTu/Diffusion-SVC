from transformers import LlamaForCausalLM, LlamaConfig
import torch
from torch import nn
from text.symbols import *

from cluster import get_cluster_model, get_cluster_result, get_cluster_center_result, get_center

from torch.nn.utils.rnn import pad_sequence, pack_sequence


class Llama(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        mode = "phone",
        semantic_kmeans_num = 10000,
        codebook_path = "pretrain/semantic_codebook.pt",
        ):
        super().__init__()
        self.mode = mode
        self.config = config
        if "phone" in self.mode:
            token_size = len(symbols)
            token_size += semantic_kmeans_num + num_tones
            self.PAD = pad_id
            self.BOS = token_size
            self.SEQ = token_size + 1
            self.EOS = token_size + 2
            token_size += 3
            # self.tone_emb = nn.Embedding(num_tones, config.hidden_size)
            # self.phone_emb = nn.Embedding(token_size + 2, config.hidden_size)
        config.vocab_size = token_size
        self.llama = LlamaForCausalLM(config)
        self.quantizer = get_cluster_model(codebook_path)

        if self.llama.model.embed_tokens.weight.data.shape[1] == self.quantizer.cluster_centers_.shape[1]:
            self.llama.model.embed_tokens.weight.data[len(symbols) - 1:len(symbols) + semantic_kmeans_num - 1] = torch.from_numpy(self.quantizer.cluster_centers_.copy())


    def forward(
        self,
        phone,
        tone,
        semantic,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        if input_ids == None:
            B,T = phone.shape
            phone_emb = self.phone_emb(phone)
            tone_emb = self.tone_emb(tone)
            seq_emb = self.phone_emb(torch.ones((B, 1),device=phone.device) * self.SEQ)
            bos_emb = self.phone_emb(torch.ones((B, 1),device=phone.device) * self.BOS)
            eos_emb = self.llama.model.embed_tokens(torch.ones_like(phone[:,0:1]) * self.EOS)
            phone_tone_emb = phone_emb + tone_emb

            phone_tone_emb = torch.cat([bos_emb, phone_tone_emb, seq_emb], dim=1)

            semantic_emb = self.llama.model.embed_tokens(semantic)
            semantic_emb = torch.cat([semantic_emb, eos_emb], dim=1)

            inputs_embeds = torch.cat([phone_tone_emb, semantic_emb], dim=1)
        else:
            outputs = self.llama(
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

if __name__ == '__main__':
    a = LlamaConfig(
         hidden_size=768,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_hidden_groups=1,
            intermediate_size=512,
    )
    b = Llama(config=a)
    phone = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
    tone = torch.LongTensor([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    semantic = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
    labels = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10,11,1,2,3,4,5,6,7,8,9,10,11,1,2,3,8,9,10,11,-100,-100,-100,-100]])
    outputs = b(phone=phone, tone=tone, semantic=semantic,labels=labels)
    print(outputs)

