import ocnn
import torch
from mmdet3d.models.builder import BACKBONES


class StemLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = ocnn.nn.OctreeConv(
            in_channels, out_channels, kernel_size=[3], stride=2, nempty=True
        )
        # self.norm = ocnn.nn.OctreeInstanceNorm(out_channels, nempty=True)
        self.norm = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()
        self.maxpool = ocnn.nn.OctreeMaxPool(nempty=True)

    def forward(self, x, octree, depth):
        x = self.conv(x, octree, depth)
        # x = self.norm(x, octree, depth - 1)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x, octree, depth - 1)
        return x, depth - 2


class ResLayer(torch.nn.Module):
    def __init__(self, block, in_channels, out_channels, blocks, stride):
        super().__init__()
        self.blocks = blocks
        if stride != 1:
            downsample = DownSample(
                in_channels, out_channels, kernel_size=[3], stride=2
            )
        else:
            downsample = None
        self.layers = torch.nn.ModuleList()
        self.layers.append(
            block(
                in_channels,
                out_channels,
                kernel_size=[3],
                stride=stride,
                downsample=downsample,
            )
        )
        for _ in range(1, blocks):
            self.layers.append(
                block(out_channels, out_channels, kernel_size=[3], stride=1)
            )

    def forward(self, x, octree, depth):
        for i in range(self.blocks):
            x, depth = self.layers[i](x, octree, depth)
        return x, depth


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = ocnn.nn.OctreeConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            nempty=True,
        )
        self.norm1 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = ocnn.nn.OctreeConv(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, nempty=True
        )
        self.norm2 = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()
        self.downsample = downsample if downsample else None

    def forward(self, x, octree, depth):
        residual = x

        out = self.conv1(x, octree, depth)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out, octree, depth if self.stride == 1 else depth - 1)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x, octree, depth)

        out += residual
        out = self.relu(out)
        return out, depth if self.stride == 1 else depth - 1


class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = ocnn.nn.OctreeConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            nempty=True,
        )
        self.norm = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x, octree, depth):
        x = self.conv(x, octree, depth)
        x = self.norm(x)
        return x


class OcnnResNetBase(torch.nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, n_outs):
        super().__init__()
        self.n_outs = n_outs
        self.inplanes = self.INIT_DIM
        self.conv1 = StemLayer(in_channels, self.inplanes)

        self.layer1 = ResLayer(
            self.BLOCK, self.inplanes, self.PLANES[0], self.LAYERS[0], stride=2
        )
        if n_outs > 1:
            self.layer2 = ResLayer(
                self.BLOCK, self.PLANES[0], self.PLANES[1], self.LAYERS[1], stride=2
            )
        if n_outs > 2:
            self.layer3 = ResLayer(
                self.BLOCK, self.PLANES[1], self.PLANES[2], self.LAYERS[2], stride=2
            )
        if n_outs > 3:
            self.layer4 = ResLayer(
                self.BLOCK, self.PLANES[2], self.PLANES[3], self.LAYERS[3], stride=2
            )

    def init_weights(self):
        pass  # each module has its own default initialization function

    def forward(self, x, octree, depth):
        outs = []
        depths = []
        x, depth = self.conv1(x, octree, depth)
        x, depth = self.layer1(x, octree, depth)
        outs.append(x)
        depths.append(depth)
        if self.n_outs == 1:
            return outs
        x, depth = self.layer2(x, octree, depth)
        outs.append(x)
        depths.append(depth)
        if self.n_outs == 2:
            return outs
        x, depth = self.layer3(x, octree, depth)
        outs.append(x)
        depths.append(depth)
        if self.n_outs == 3:
            return outs
        x, depth = self.layer4(x, octree, depth)
        outs.append(x)
        depths.append(depth)
        return outs, depths


@BACKBONES.register_module()
class OcnnResNet3D(OcnnResNetBase):
    def __init__(self, in_channels, depth, n_outs=4):
        if depth == 14:
            self.BLOCK = ResBlock
            self.LAYERS = (1, 1, 1, 1)
        elif depth == 18:
            self.BLOCK = ResBlock
            self.LAYERS = (2, 2, 2, 2)
        elif depth == 34:
            self.BLOCK = ResBlock
            self.LAYERS = (3, 4, 6, 3)
        elif depth == 50:
            self.BLOCK = ResBlock
            self.LAYERS = (4, 3, 6, 3)
        elif depth == 101:
            self.BLOCK = ResBlock
            self.LAYERS = (3, 4, 23, 3)
        else:
            raise ValueError(f"invalid depth={depth}")

        super().__init__(in_channels, n_outs)
