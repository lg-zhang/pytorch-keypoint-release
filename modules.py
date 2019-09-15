import torch
import torch.nn as nn
import torch.nn.modules.loss as L
import math


def conv2d_relu(iplanes, oplanes, ksize):
    return [
        nn.Conv2d(
            in_channels=iplanes, out_channels=oplanes, kernel_size=ksize
        ),
        nn.ReLU(True),
    ]


def make_fcn_nopool65():
    # 65 - 8 - 6 * 8 - 8
    layers = []
    # 65 - 8 = 57
    layers += conv2d_relu(1, 16, 9)
    # 57 - 6 = 51
    layers += conv2d_relu(16, 32, 7)
    # 51 - 6 * 7 = 9
    for _ in range(7):
        layers += conv2d_relu(32, 32, 7)
    # 9 - 8 = 1
    layers += conv2d_relu(32, 32, 9)
    # fc-1
    layers += conv2d_relu(32, 32, 1)
    # fc-2
    layers += [
        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0)
    ]

    return nn.Sequential(*layers)


class KeypointNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.net = make_fcn_nopool65()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                # nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.net(x)
        if self.training:
            return x.view(-1, x.size(2) * x.size(3))
        return x

    def get_output_size(self, w, h):
        return w - 64, h - 64, 32, 32


class LossFunction(nn.Module):
    def __init__(
        self,
        margin=1.0,
        optimize_maxima=False,
        topk=10,
        peakedness_margin=3.0,
        peakedness_weight=0.0,
    ):
        super().__init__()
        self._margin = margin
        self._optimize_maxima = optimize_maxima
        self._topk = topk
        self._peakedness_margin = peakedness_margin
        self._peakedness_weight = peakedness_weight

    def _compute_peakedness_loss(self, score):
        sorted_score, _ = score.sort(dim=1, descending=self._optimize_maxima)
        score_extrema = sorted_score[:, 0]
        _, idx = score_extrema.sort(descending=self._optimize_maxima)

        sorted_score = sorted_score[idx[0:int(len(idx) * 0.75)].data, :]
        num = min(sorted_score.size()[1], self._topk)
        aac = (
            torch.abs(
                sorted_score[:, 0].contiguous().view(-1, 1).repeat(1, num)
                - sorted_score[:, 0:num]
            )
        ).mean(dim=1)

        return (self._peakedness_margin - aac).clamp(0)

    def forward(self, sa1, sa2, sb1, sb2):
        c = sa1.size(1) // 2  # only use the center
        diff_a = sa1[:, c] - sa2[:, c]
        diff_b = sb1[:, c] - sb2[:, c]
        diff_prod = diff_a * diff_b
        loss_tensor = (self._margin - diff_prod).clamp(0)

        loss = loss_tensor.mean()

        if self._peakedness_weight > 0:
            loss += (
                (
                    self._compute_peakedness_loss(sa1)
                    + self._compute_peakedness_loss(sa2)
                    + self._compute_peakedness_loss(sb1)
                    + self._compute_peakedness_loss(sb2)
                )
                / 4.0
                * self._peakedness_weight
            )

        return loss
