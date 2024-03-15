from vq_ae.wavenet.wavenet import WN
from vector_quantize_pytorch import VectorQuantize
from torch import nn
import torch.nn.functional as F
import torch

class VQCNN(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, num_codebook):
        super(VQCNN, self).__init__()
        self.transformer_encoder = WN(
            hidden_channels=d_model,
            kernel_size=5,
            dilation_rate=1,
            n_layers=num_layers,
            gin_channels=0,
        )
        
        self.quantizer = VectorQuantize(
            dim = d_model,
            codebook_size = num_codebook,
            codebook_dim = 32,
            decay = 0.8,             
            commitment_weight = 1.,
            use_cosine_sim=True
        )

        self.transformer_decoder = WN(
            hidden_channels=d_model,
            kernel_size=5,
            dilation_rate=1,
            n_layers=num_layers,
            gin_channels=0,
        )

    def forward(self, units, **kwargs):
        training = kwargs.get('training', self.training)
        mask = kwargs.get('mask', None)
        if mask is not None:
            mask = mask[:, None, :]
        else:
            mask = torch.ones(units.shape[0], 1, units.shape[1]).to(units.device)
        if training:
            x = self.transformer_encoder(units.transpose(1, 2), mask)
            x = x.transpose(1, 2)
            x, indices, commit_loss = self.quantizer(x)
            # 如果indices所有的数相同，打印indices
            x = x.transpose(1, 2)
            tgt = self.transformer_decoder(x, mask)
            tgt = tgt.transpose(1, 2)
            l1_loss = F.smooth_l1_loss(tgt, units, reduction = 'none')
            if mask is not None:
                l1_loss = l1_loss.sum() / mask.sum()
            else:
                l1_loss = l1_loss.mean()
            return l1_loss, commit_loss
        else:
            x = self.transformer_encoder(units.transpose(1, 2), mask).transpose(1, 2)
            x, indices, commit_loss = self.quantizer(x)
            return x, indices, commit_loss
    
    def set_eval_mode(self):
        self.transformer_decoder = self.transformer_decoder.cpu()
        del self.transformer_decoder
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

def get_model(args):
    return VQCNN(
        d_model = args.data.encoder_out_channels,
        nhead = args.vqae.n_heads,
        dim_feedforward = args.data.encoder_out_channels * 2,
        num_layers = args.vqae.n_layers,
        num_codebook = args.model.text2semantic.semantic_kmeans_num
    )

if __name__ == '__main__':
    model = VQCNN(512, 8, 2048, 6, 512)
    src = torch.rand(10, 32, 512)
    l1_loss, commit_loss = model(src)
    print(l1_loss, commit_loss)
        