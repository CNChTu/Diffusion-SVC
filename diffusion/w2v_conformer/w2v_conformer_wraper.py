from diffusion.w2v_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerModel, Wav2Vec2ConformerConfig
import torch
from torch.nn import Embedding
class Wav2Vec2ConformerWrapper(torch.nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super().__init__()
        config.position_embeddings_type = None
        self.model = Wav2Vec2ConformerModel(config)
        self.timestep_embedding = Embedding(1000, config.hidden_size)
        self.input_proj = torch.nn.Linear(in_channels, config.hidden_size)
        self.output_proj = torch.nn.Linear(config.hidden_size, out_channels)

    def forward(self, input_values, timestep, attention_mask=None):
        timestep = timestep.to(torch.long)
        input_values = input_values.transpose(1, 2)
        x = self.input_proj(input_values)
        timestep_emb = self.timestep_embedding(timestep)
        x = x + timestep_emb[:,None,:]
        x = self.model(x, timestep_emb, attention_mask=attention_mask).last_hidden_state
        x = self.output_proj(x).transpose(1, 2)
        return Wav2Vec2ConformerWrapperOutPut(x)

class Wav2Vec2ConformerWrapperOutPut:
    def __init__(self, sample):
        self.sample = sample

if __name__ == "__main__":
    config = Wav2Vec2ConformerConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        feat_proj_dropout=0.0,
        feat_quantizer_dropout=0.0,
        final_dropout=0.1,
        layerdrop=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        feat_extract_norm="group",
        feat_extract_activation="gelu",
        conv_dim=(256, 512, 512, 512),
        conv_stride=(5, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3),
        conv_bias=False
    )
    wrapper = Wav2Vec2ConformerWrapper(376, 64, config)
    input_values = torch.rand(2, 250, 376)
    timestep = torch.randint(1000, (2, ))
    print(timestep)
    output = wrapper(input_values, timestep)
    print(output.shape)
    print(output)