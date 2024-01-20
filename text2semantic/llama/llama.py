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
        use_flash_attn = False,
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
            self.num_tones = num_tones
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

        config.bos_token_id = semantic_kmeans_num
        config.eos_token_id = semantic_kmeans_num + 1
        config.pad_token_id = semantic_kmeans_num + 2
        token_size += semantic_kmeans_num + 3

        config.vocab_size = token_size
        # llama用flash attention 2.0
        if use_flash_attn:
            config._attn_implementation = "flash_attention_2"
        self.llama = LlamaForCausalLM(config=config)
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
            if self.mode == "phone":
                phone = torch.cat([torch.tensor([[self.BOS]]), phone, torch.tensor([[self.EOS]])], dim=1)
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
    labels = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10,11,1,2,3,4,5,6,7,8,9,10,11,1,2,3,8,9,10,11,-100,-100,-100,-100,-100]])
    outputs = b(phone=phone, tone=tone, semantic=semantic,labels=labels)
    print(outputs)

