from torch import nn
import torch

class ConformerConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(self,
                 hidden_size,
                 conv_depthwise_kernel_size
                 ):
        super().__init__()
        if (conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pointwise_conv1 = nn.Conv1d(
            hidden_size,
            2 * hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            conv_depthwise_kernel_size,
            stride=1,
            padding=0,
            groups=hidden_size,
            bias=False,
        )

        self.depthwise_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        hidden_states_input = hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.layer_norm(hidden_states)

        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.transpose(1, 2)

        # GLU mechanism
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # Pad the sequence entirely on the left because of causal convolution.
        hidden_states = torch.nn.functional.pad(hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0))

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)

        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # hidden_states = hidden_states.transpose(1, 2)
        return hidden_states + hidden_states_input * 0.5
    
class ConformerConvLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 conv_depthwise_kernel_size,
                 layer_num
                 ):
        super().__init__()
        self.lzyers = nn.ModuleList([
            ConformerConvolutionModule(
                hidden_size,
                conv_depthwise_kernel_size
            ) for _ in range(layer_num)
        ])
    def forward(self, hidden_states):
        for layer in self.lzyers:
            hidden_states = layer(hidden_states)
        return hidden_states
    