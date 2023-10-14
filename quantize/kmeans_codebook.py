from torch import nn
import torch
from einops import rearrange
from torch.nn import functional as F

class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        codebook_weight,
    ):
        super().__init__()

        self.register_buffer("embed", torch.tensor(codebook_weight))

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize
    
    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x
    
    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        embed_ind = self.encode(x)
        return self.decode(embed_ind)