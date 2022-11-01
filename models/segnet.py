from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from .modules import ConvBnReLU2d
from collections import deque


def segnet(in_channels, out_channels):
    return SegNet(in_channels, out_channels)


class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()

        def make_layer(in_channels, out_channels, num_blocks):
            layers = [ConvBnReLU2d(in_channels, out_channels, 3, padding=1)]
            for _ in range(1, num_blocks):
                layers += [ConvBnReLU2d(out_channels, out_channels, 3, padding=1)]
            return nn.Sequential(*layers)

        self.encoder = nn.ModuleDict(OrderedDict([
            ('layer1', make_layer(in_channels, 64, 2)),
            ('layer2', make_layer(64, 128, 2)),
            ('layer3', make_layer(128, 256, 3)),
            ('layer4', make_layer(256, 512, 3)),
            ('layer5', make_layer(512, 512, 3)),
        ]))

        self.decoder = nn.ModuleDict(OrderedDict([
            ('layer1', make_layer(512, 512, 3)),
            ('layer2', make_layer(512, 256, 3)),
            ('layer3', make_layer(256, 128, 3)),
            ('layer4', make_layer(128, 64, 2)),
            ('layer5', make_layer(64, out_channels, 2)),
        ]))

    def forward(self, input):
        pooling_indices = []
        x, mp_sizes = input, deque()
        for i, layer in enumerate(self.encoder.children()):
            x = layer(x)
            if i == 0:
                mp_sizes.append(x.size())
            x, indices = F.max_pool2d_with_indices(x, kernel_size=2)
            mp_sizes.appendleft(x.shape)
            pooling_indices = [indices, *pooling_indices]
        mp_sizes.popleft()
        for i, layer in enumerate(self.decoder.children()):
            indices, *pooling_indices = pooling_indices
            x = F.max_unpool2d(x, indices, kernel_size=2, output_size=mp_sizes[i])
            x = layer(x)
        return x
    
    
if __name__ == "__main__":
    from ..constants.config import *
    import torch
    
    segmenter = SegNet(INPUT_CHANNELS, NUM_CLASSES)
    inpt_dum = torch.rand(BATCH_SIZE, INPUT_CHANNELS, *INPUT_SHAPE)
    out = segmenter(inpt_dum)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES, *INPUT_SHAPE)
    print("Model passed!")