import torch
import torch.nn as nn
import torch.nn.functional as F

from .stable_rank_utils import stable_rank


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, spectral_norm=False, stable=False, rank=0.1):
        super().__init__()

        if spectral_norm:
            spn = nn.utils.spectral_norm
        else:
            spn = lambda x: x

        self.bn1 = spn(nn.BatchNorm2d(in_planes))
        self.conv1 = spn(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        )
        self.bn2 = spn(nn.BatchNorm2d(planes))
        self.conv2 = spn(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))

        if stable:
            self.conv1 = stable_rank(self.conv1, rank=rank)
            self.conv2 = stable_rank(self.conv2, rank=rank)

        if stride != 1 or in_planes != self.expansion * planes:
            downsample_conv = spn(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                )
            )
            if stable:
                downsample_conv = stable_rank(downsample_conv, rank=rank)
            self.shortcut = nn.Sequential(downsample_conv)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBlockWithoutBN(nn.Module):
    """Pre-activation version of the BasicBlock without BN."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, stable=False, rank=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stable:
            self.conv1 = stable_rank(self.conv1, rank=rank)
            self.conv2 = stable_rank(self.conv2, rank=rank)

        if stride != 1 or in_planes != self.expansion * planes:
            downsample_conv = spn(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                )
            )
            if stable:
                downsample_conv = stable_rank(downsample_conv, rank=rank)
            self.shortcut = nn.Sequential(downsample_conv)

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(x) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, stable=False, rank=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        if stable:
            self.conv1 = stable_rank(self.conv1, rank=rank)
            self.conv2 = stable_rank(self.conv2, rank=rank)
            self.conv3 = stable_rank(self.conv3, rank=rank)

        if stride != 1 or in_planes != self.expansion * planes:
            downsample_conv = spn(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                )
            )
            if stable:
                downsample_conv = stable_rank(downsample_conv, rank=rank)
            self.shortcut = nn.Sequential(downsample_conv)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        spectral_norm=False,
        stable=False,
        rank=0.1,
        num_channels=3,
    ):
        super().__init__()
        if spectral_norm:
            spn = nn.utils.spectral_norm
        else:
            spn = lambda x: x
        self.in_planes = 64
        self.conv1 = spn(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )
        if stable:
            print(f"Using stable rank with rank {rank}")
            self.conv1 = stable_rank(self.conv1, rank=rank)
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            spectral_norm=spectral_norm,
            stable=stable,
            rank=rank,
        )
        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            spectral_norm=spectral_norm,
            stable=stable,
            rank=rank,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            spectral_norm=spectral_norm,
            stable=stable,
            rank=rank,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            spectral_norm=spectral_norm,
            stable=stable,
            rank=rank,
        )
        self.bn = spn(nn.BatchNorm2d(512 * block.expansion))
        self.linear = spn(nn.Linear(512 * block.expansion, num_classes))

    def _make_layer(self, block, planes, num_blocks, stride, spectral_norm, stable, rank):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, spectral_norm, stable, rank))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(spectral_norm=False, **kwargs):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], spectral_norm=spectral_norm, **kwargs)


def PreActResNet18_withoutBN(**kwargs):
    return PreActResNet(PreActBlockWithoutBN, [2, 2, 2, 2], **kwargs)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
