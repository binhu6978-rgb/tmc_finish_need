import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_prefix_gate(x, ratio, tau):
    """Differentiable prefix gate used only for capacity feedback."""
    if not torch.is_tensor(ratio):
        ratio = x.new_tensor(float(ratio))
    ratio = ratio.to(device=x.device, dtype=x.dtype)
    tau = max(float(tau), 1e-4)
    channels = x.shape[1]
    # Paper definition: sigmoid((m*C-c+1/2)/tau), c in {1,...,C}.
    # Keeping tau in channel-index units also makes m=1 approximate the full
    # hard stage while preserving a valid gradient for strong clients.
    positions = torch.arange(
        channels, device=x.device, dtype=x.dtype) + 0.5
    gate = torch.sigmoid(
        (ratio * channels - positions) / tau).view(1, channels, 1, 1)
    return x * gate


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, track_running_stats=True):
        super(ResidualBlock, self).__init__()
        inchannel = inchannel
        outchannel = outchannel
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=track_running_stats)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:  # 这两个东西是在说一码事，再升维的时候需要增大stride来保持计算量
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel, track_running_stats=track_running_stats)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)  # 当shortcut是 nn.Sequential()的时候 会返回x本身
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_channels=3, num_classes=10,
                 track_running_stats=True, slim_idx=0, scale=1.0,
                 dataset='cifar', stage_scales=None):
        super(ResNet, self).__init__()

        self.dataset = dataset
        self.stage_scales = None if stage_scales is None else [
            float(max(0.05, min(1.0, s))) for s in stage_scales
        ]
        if self.stage_scales is not None and len(self.stage_scales) != 3:
            raise ValueError("stage_scales must contain [layer2, layer3, layer4]")
        if self.dataset == 'widar':
            self.reshape = nn.Sequential(
                nn.ConvTranspose2d(22, num_channels, 7, stride=1),
                nn.ReLU(),
                nn.ConvTranspose2d(num_channels, num_channels, kernel_size=7, stride=1),
                nn.ReLU()
            )

        idx = 0
        if self.stage_scales is not None:
            self.inchannel = 64
        elif idx < slim_idx:
            self.inchannel = 64
        else:
            self.inchannel = int(64 * scale)

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel, track_running_stats=track_running_stats),
            nn.ReLU()
        )

        idx += 1
        tmp_scale = 1.0 if self.stage_scales is not None else (1.0 if idx < slim_idx else scale)
        self.layer1 = self._make_layer(ResidualBlock, int(96 * tmp_scale), 2, stride=1, track_running_stats=track_running_stats)

        idx += 1
        tmp_scale = self.stage_scales[0] if self.stage_scales is not None else (1.0 if idx < slim_idx else scale)
        self.layer2 = self._make_layer(ResidualBlock, int(128 * tmp_scale), 2, stride=2, track_running_stats=track_running_stats)

        idx += 1
        tmp_scale = self.stage_scales[1] if self.stage_scales is not None else (1.0 if idx < slim_idx else scale)
        self.layer3 = self._make_layer(ResidualBlock, int(256 * tmp_scale), 2, stride=2, track_running_stats=track_running_stats)

        idx += 1
        tmp_scale = self.stage_scales[2] if self.stage_scales is not None else (1.0 if idx < slim_idx else scale)
        self.layer4 = self._make_layer(ResidualBlock, int(512 * tmp_scale), 2, stride=2, track_running_stats=track_running_stats)

        idx += 1
        tmp_scale = self.stage_scales[2] if self.stage_scales is not None else (1.0 if idx < slim_idx else scale)
        self.fc = nn.Linear(int(512 * tmp_scale), num_classes)
        idx += 1

    def _make_layer(self, block, channels, num_blocks, stride, track_running_stats):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, track_running_stats))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'widar':
            out = self.conv1(self.reshape(x))
        else:
            out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        result = {'representation': out}
        out = F.adaptive_avg_pool2d(out, (1, 1))  # ← 只改这一行
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        result['output'] = out
        return result

    def forward_soft(self, x, capacity, tau=0.05):
        """Full-width differentiable proxy for d(loss)/d(capacity).

        Actual client training still uses physically narrow submodels. This
        path is called only on a full-width proxy during control rounds.
        """
        if len(capacity) != 3:
            raise ValueError("capacity must contain three stage ratios")
        if self.dataset == 'widar':
            out = self.conv1(self.reshape(x))
        else:
            out = self.conv1(x)
        out = self.layer1(out)
        out = _soft_prefix_gate(self.layer2(out), capacity[0], tau)
        out = _soft_prefix_gate(self.layer3(out), capacity[1], tau)
        out = _soft_prefix_gate(self.layer4(out), capacity[2], tau)
        representation = out
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        out = self.fc(out)
        return {'representation': representation, 'output': out}

def ResNet18_cifar(num_channels=3, num_classes=10, track_running_stats=True,
                   slim_idx=0, scale=1.0, stage_scales=None):
    return ResNet(ResidualBlock, num_channels, num_classes, track_running_stats,
                  slim_idx, scale, 'cifar', stage_scales)  # 当前结构为 [3,4,6,3]，即 ResNet34


def ResNet18_widar(num_channels=3, num_classes=22, track_running_stats=True,
                   slim_idx=0, scale=1.0, stage_scales=None):
    return ResNet(ResidualBlock, num_channels, num_classes, track_running_stats,
                  slim_idx, scale, 'widar', stage_scales)


if __name__ == '__main__':
    net_1 = ResNet18_cifar(num_classes=10, track_running_stats=True, slim_idx=2, scale=0.45)
    #     net_2 = ResNet18(num_classes=10, track_running_stats=False, slim_idx=2, scale=0.75)
    print(net_1)
