from transformers.models.bert.modeling_bert import BertEncoder, BertConfig
from vector_quantize_pytorch import VectorQuantize
from torch import nn
import torch.nn.functional as F
import torch

class VQTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, num_codebook):
        super(VQTransformer, self).__init__()
        config = BertConfig(
            hidden_size = d_model,
            num_hidden_layers = num_layers,
            num_attention_heads = nhead,
            intermediate_size = dim_feedforward
        )


        self.transformer_encoder = BertEncoder(config)
        
        self.quantizer = VectorQuantize(
            dim = d_model,
            codebook_size = num_codebook,
            codebook_dim = 32,
            decay = 0.8,             
            commitment_weight = 1.,
            use_cosine_sim=True
        )

        self.transformer_decoder = BertEncoder(config)

    def forward(self, units, **kwargs):
        training = kwargs.get('training', self.training)
        mask = kwargs.get('mask', None)
        if mask is not None:
            mask = mask[:, None, None, :]
        if training:
            x = self.transformer_encoder(hidden_states = units, attention_mask = mask).last_hidden_state
            if mask is not None:
                x = x * mask[:, 0, 0, :, None]
            x, indices, commit_loss = self.quantizer(x)
            tgt = self.transformer_decoder(hidden_states = units, attention_mask = mask).last_hidden_state
            if mask is not None:
                mask = mask[:, 0, 0, :]
                tgt = tgt * mask[:, :, None]
            l1_loss = F.smooth_l1_loss(tgt, units, reduction = 'none')
            if mask is not None:
                l1_loss = l1_loss.sum() / mask.sum()
            else:
                l1_loss = l1_loss.mean()
            return l1_loss, commit_loss
        else:
            x = self.transformer_encoder(hidden_states = units).last_hidden_state
            x, indices, commit_loss = self.quantizer(x)
            return x, indices, commit_loss
    
    def set_eval_mode(self):
        self.transformer_decoder = self.transformer_decoder.cpu()
        del self.transformer_decoder
        self.eval()

def get_model(args):
    return VQTransformer(
        d_model = args.data.encoder_out_channels,
        nhead = 4,
        dim_feedforward = args.data.encoder_out_channels * 2,
        num_layers = 3,
        num_codebook = args.model.text2semantic.semantic_kmeans_num
    )

if __name__ == '__main__':
    model = VQTransformer(512, 8, 2048, 6, 512)
    src = torch.rand(10, 32, 512)
    l1_loss, commit_loss = model(src)
    print(l1_loss, commit_loss)
        