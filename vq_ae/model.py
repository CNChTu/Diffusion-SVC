from torch.nn import TransformerEncoderLayer, TransformerEncoder
from vector_quantize_pytorch import VectorQuantize
from torch import nn
import torch.nn.functional as F
import torch

class VQTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, num_codebook):
        super(VQTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, 0.)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        self.quantizer = VectorQuantize(
            dim = d_model,
            codebook_size = num_codebook,
            codebook_dim = 32,
            decay = 0.8,             
            commitment_weight = 1.,
            use_cosine_sim=True
        )

        self.transformer_decoder = TransformerEncoder(encoder_layers, num_layers)
    
    def forward(self, units, **kwargs):
        x = self.transformer_encoder(units)
        x, indices, commit_loss = self.quantizer(x)
        tgt = self.transformer_decoder(x)
        l1_loss = F.smooth_l1_loss(tgt, units)
        return l1_loss, commit_loss

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
        