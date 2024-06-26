import logging
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)

class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        if hasattr(model, "module"):
            self.model_state_dict = deepcopy(model.module.state_dict())
        else:
            self.model_state_dict = deepcopy(model.state_dict())
        self.decay = decay
        self.device = device  # perform ema on different device from model if set

    def _update(self, model, update_fn):
        model_values = model.module.state_dict().values() if hasattr(model, "module") else model.state_dict().values()
        with torch.no_grad():
            for ema_v, model_v in zip(self.model_state_dict.values(), model_values):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model): # 使用衰减率更新 EMA 参数
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):  # 直接将 EMA 参数设置为与提供的模型参数相同。
        self._update(model, update_fn=lambda e, m: m)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model_state_dict
