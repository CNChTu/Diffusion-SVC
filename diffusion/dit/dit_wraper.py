from diffusion.dit.model import DiT
import torch

class DiTModel(torch.nn.Module):
    def __init__(self,
        in_channels=256,
        hidden_size=768,
        out_channels=768,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        attention_dropout=0.0,
        project_dropout=0.0,
    ):
        super().__init__()
        self.model = DiT(
        in_channels=in_channels,
        hidden_size=hidden_size,
        out_channels=out_channels,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        attention_dropout=attention_dropout,
        project_dropout=project_dropout)

    def forward(self, input_values, timestep):
        timestep = timestep.to(torch.long)
        input_values = input_values.transpose(1, 2)
        x = self.model(input_values, timestep).transpose(1, 2)
        return Wav2Vec2ConformerWrapperOutPut(x)

class Wav2Vec2ConformerWrapperOutPut:
    def __init__(self, sample):
        self.sample = sample