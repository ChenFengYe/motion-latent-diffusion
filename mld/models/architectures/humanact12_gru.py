import torch
import torch.nn as nn


# adapted from action2motion to take inputs of different lengths
class MotionDiscriminator(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 hidden_layer,
                 output_size=12,
                 use_noise=None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise

        self.recurrent = nn.GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = nn.Linear(hidden_size, 30)
        self.linear2 = nn.Linear(30, output_size)

    def forward(self, motion_sequence, lengths=None, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        bs, njoints, nfeats, num_frames = motion_sequence.shape
        motion_sequence = motion_sequence.reshape(bs, njoints * nfeats,
                                                  num_frames)
        motion_sequence = motion_sequence.permute(2, 0, 1)
        if hidden_unit is None:
            # motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1),
                                          self.hidden_layer).to(
                                              motion_sequence.device)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)

        # select the last valid, instead of: gru_o[-1, :, :]
        out = gru_o[tuple(
            torch.stack(
                (lengths - 1, torch.arange(bs,
                                           device=motion_sequence.device))))]

        # dim (num_samples, 30)
        lin1 = self.linear1(out)
        lin1 = torch.tanh(lin1)
        # dim (num_samples, output_size)
        lin2 = self.linear2(lin1)
        return lin2

    def initHidden(self, num_samples, layer):
        return torch.randn(layer,
                           num_samples,
                           self.hidden_size,
                           requires_grad=False)


class MotionDiscriminatorForFID(MotionDiscriminator):

    def forward(self, motion_sequence, lengths=None, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        bs, njoints, nfeats, num_frames = motion_sequence.shape
        motion_sequence = motion_sequence.reshape(bs, njoints * nfeats,
                                                  num_frames)
        motion_sequence = motion_sequence.permute(2, 0, 1)
        if hidden_unit is None:
            # motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1),
                                          self.hidden_layer).to(
                                              motion_sequence.device)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)

        # select the last valid, instead of: gru_o[-1, :, :]
        out = gru_o[tuple(
            torch.stack(
                (lengths - 1, torch.arange(bs,
                                           device=motion_sequence.device))))]

        # dim (num_samples, 30)
        lin1 = self.linear1(out)
        lin1 = torch.tanh(lin1)
        return lin1
