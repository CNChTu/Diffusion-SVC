import math
import torch
from torch.nn import functional as F

def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if "Depthwise_Separable" in classname:
    m.depth_conv.weight.data.normal_(mean, std)
    m.point_conv.weight.data.normal_(mean, std) 
  elif classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
  return int((kernel_size*dilation - dilation)/2)

def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
  position = torch.arange(length, dtype=torch.float)
  num_timescales = channels // 2
  log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1))
  inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment)
  scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
  signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
  signal = F.pad(signal, [0, 0, 0, channels % 2])
  signal = signal.view(1, channels, length)
  return signal

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  return acts

def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)
  if clip_value is not None:
    clip_value = float(clip_value)

  total_norm = 0
  for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type
    if clip_value is not None:
      p.grad.data.clamp_(min=-clip_value, max=clip_value)
  total_norm = total_norm ** (1. / norm_type)
  return total_norm
