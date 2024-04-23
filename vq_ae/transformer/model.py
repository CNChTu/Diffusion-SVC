from transformers.models.bert.modeling_bert import BertEncoder, BertConfig
from vector_quantize_pytorch import VectorQuantize
from torch import nn
import torch.nn.functional as F
import torch

class VQTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, num_codebook, units_scale, time_downsample_rate=1):
        super(VQTransformer, self).__init__()
        self.time_downsample_rate = time_downsample_rate if time_downsample_rate is not None else 1
        if time_downsample_rate is not None:
            self.time_downsample = nn.Conv1d(d_model, d_model, time_downsample_rate * 2, stride = time_downsample_rate, padding= (time_downsample_rate + 1) // 2)
            self.time_upsample = nn.ConvTranspose1d(d_model, d_model, time_downsample_rate * 2, stride = time_downsample_rate, padding = time_downsample_rate // 2)
        else:
            self.time_downsample = nn.Identity()
            self.time_upsample = nn.Identity()
        
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
        self.units_scale = units_scale

    def forward(self, units, **kwargs):
        if self.time_downsample_rate != 1 and units.shape[-2]%self.time_downsample_rate != 0:
            units = units[:, :-(units.shape[-2]%self.time_downsample_rate), :]
        units = units/self.units_scale
        training = kwargs.get('training', self.training)
        mask = kwargs.get('mask', None)
        if mask is not None:
            mask = mask[:, None, None, :]
        if self.time_downsample_rate > 1:
            if mask.shape[-1]%self.time_downsample_rate != 0:
                mask = mask[:, :, :, :-(mask.shape[-1]%self.time_downsample_rate)]
            mask = mask[:, :, :, ::self.time_downsample_rate]
        x = self.time_downsample(units.transpose(1, 2)).transpose(1, 2)
        if training:
            x = self.transformer_encoder(hidden_states = units, attention_mask = mask).last_hidden_state
            if mask is not None:
                x = x * mask[:, 0, 0, :, None]
            x, indices, commit_loss = self.quantizer(x)
            tgt = self.transformer_decoder(hidden_states = x, attention_mask = mask).last_hidden_state
            tgt = self.time_upsample(tgt.transpose(1, 2)).transpose(1, 2)
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
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

def get_model(args):
    return VQTransformer(
        d_model = args.data.encoder_out_channels,
        nhead = args.vqae.n_heads,
        dim_feedforward = args.data.encoder_out_channels * 2,
        num_layers = args.vqae.n_layers,
        num_codebook = args.model.text2semantic.semantic_kmeans_num,
        units_scale = args.vqae.units_scale,
        time_downsample_rate = args.vqae.time_downsample_rate
    )

if __name__ == '__main__':
    model = VQTransformer(512, 8, 2048, 6, 512)
    src = torch.rand(10, 32, 512)
    l1_loss, commit_loss = model(src)
    print(l1_loss, commit_loss)
        