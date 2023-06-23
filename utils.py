from __future__ import annotations

from dataclasses import dataclass
from torch.autograd import Variable
import torch


@dataclass
class StatTracker:
    avg: float = 0
    sum: float = 0
    cnt: float = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def get_output_shape(
    *layers: torch.nn.Sequential | torch.nn.Module,
    shape: tuple[int, int, int],
) -> int:
    """Calculate the output dimensions of a stack of conv layer"""
    channels, w, h = shape
    input = Variable(torch.rand(1, channels, w, h))

    seq = torch.nn.Sequential()
    for layer in layers:
        seq.append(layer)

    output_feat = seq(input)

    # Flatten the data out, and get it's size, this will be
    # the size of what's given to a fully connected layer
    n_size = output_feat.data.view(1, -1).size(1)
    return n_size


# Can generate a grid like this in order to evaluate baseline and check Pareto front
# However, given the dimensionality of the search space, better to use random search

# num_dict = {
#     'batch_size': 3,
#     'dropout_rate': 0.2,
#     'learning_rate_init': 10,
#     'n_channels_conv_0': 3,
#     'n_channels_fc_0': 3,
#     'n_channels_fc_1': 3,
#     'n_channels_fc_2': 3,
#     'n_conv_layers': 3,
#     'n_fc_layers': 3,
#     'n_channels_conv_1': 3,
#     'n_channels_conv_2': 3
# }
# import ConfigSpace
# ConfigSpace.util.generate_grid(cs, num_dict)
