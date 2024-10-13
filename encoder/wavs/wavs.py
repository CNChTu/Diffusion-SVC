import torch


class WAVS(torch.nn.Module):
    def __init__(self, hop_length):
        super(WAVS, self).__init__()
        self.hop_length = hop_length

    def forward(self, x):
        # x: (batch_size, x_len)
        x_len = x.shape[-1]
        # pad x to make it a multiple of hop_length
        pad_len = (self.hop_length - x_len % self.hop_length)
        # pad zeros on the right for all batches
        x = torch.nn.functional.pad(x, (0, pad_len))
        # unfold x
        x = x.unfold(-1, self.hop_length, self.hop_length)  # (batch_size, time, hop_length)
        return x


class IWAVS(torch.nn.Module):
    def __init__(self, hop_length):
        super(IWAVS, self).__init__()
        self.hop_length = hop_length

    def forward(self, x):
        assert x.shape[-1] == self.hop_length
        # x: (batch_size, time, hop_length)
        # fold x
        x = x.flatten(-2, -1)
        return x  # (batch_size, x_len)
