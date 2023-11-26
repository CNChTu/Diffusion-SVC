from funasr.bin.asr_infer import Speech2Text
import torch
from typing import Union
import numpy as np
from funasr.torch_utils.device_funcs import to_device

class SpeechEncoder(Speech2Text):
    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None
    ):
        """Inference

        Args:
            speech: Input speech data
        Returns:
            enc: Encoded speech

        """

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if self.frontend is not None:
            feats, feats_len = self.frontend.forward(speech, speech_lengths)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
            self.asr_model.frontend = None
        else:
            feats = speech
            feats_len = speech_lengths
        
        batch = {"speech": feats, "speech_lengths": feats_len}
        print(feats.shape)

        # a. To device
        batch = to_device(batch, device=self.device)
        # b. Forward Encoder
        enc, _ = self.asr_model.encode(**batch)
        if isinstance(enc, tuple):
            enc = enc[0]
        assert len(enc) == 1, len(enc)

        return enc