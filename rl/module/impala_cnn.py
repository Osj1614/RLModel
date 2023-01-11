import torch as th

from torch import nn, Tensor
from torch.nn import functional as F

class ImpalaEncoder(nn.Module):
    def __init__(
        self,
        inshape,
        outsize=256,
        chans=(16, 32, 32),
        scale_ob=255.0,
        nblock=2,
        **kwargs
    ):
        super().__init__()
        self.cnn = ImpalaCNN(
            inshape=inshape,
            chans=chans,
            scale_ob=scale_ob,
            nblock=nblock,
            outsize=outsize,
            **kwargs
        )
        self.out_size = outsize

    def forward(self, x):
        x = self.cnn(x)
        return x


class CNNExtractor(nn.Module):
    def __init__(self, obs_shape):
        super().__init__(obs_shape)
        self.act_fn = F.relu
        assert obs_shape == (64, 64, 3)
        self.cnn = ImpalaEncoder(self.obs_shape)
        self.in_shape = obs_shape
        self.out_size = self.cnn.out_size
        
    def forward(self, input):
        cur = self.cnn(input)       
        return cur


class CnnBasicBlock(nn.Module):
    def __init__(self, inchan, batch_norm=False):
        super().__init__()
        self.inchan = inchan
        self.batch_norm = batch_norm
        self.conv0 = nn.Conv2d(self.inchan, self.inchan, 3, padding=1)
        self.conv1 = nn.Conv2d(self.inchan, self.inchan, 3, padding=1)
        if self.batch_norm:
            self.bn0 = nn.BatchNorm2d(self.inchan)
            self.bn1 = nn.BatchNorm2d(self.inchan)

    def residual(self, x):
        if self.batch_norm:
            x = self.bn0(x)
        x = F.relu(x, inplace=False)
        x = self.conv0(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        return x

    def forward(self, x):
        return x + self.residual(x)


class CnnDownStack(nn.Module):
    def __init__(self, inchan, nblock, outchan, pool=True, **kwargs):
        super().__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool
        self.firstconv = nn.Conv2d(inchan, outchan, 3, padding=1)
        self.blocks = nn.ModuleList(
            [CnnBasicBlock(outchan, **kwargs) for _ in range(nblock)]
        )

    def forward(self, x):
        x = self.firstconv(x)
        if getattr(self, "pool", True):
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if getattr(self, "pool", True):
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self.outchan, h, w)


def intprod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def sequential(layers, x, *args, diag_name=None):
    for (i, layer) in enumerate(layers):
        x = layer(x, *args)
    return x


class ImpalaCNN(nn.Module):
    name = "ImpalaCNN"  # put it here to preserve pickle compat

    def __init__(
        self, inshape, chans, outsize, scale_ob, nblock, final_relu=True, **kwargs
    ):
        super().__init__()
        self.scale_ob = scale_ob
        h, w, c = inshape
        curshape = (c, h, w)
        self.stacks = nn.ModuleList()
        for outchan in chans:
            stack = CnnDownStack(
                curshape[0], nblock=nblock, outchan=outchan, **kwargs
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        self.dense = nn.Linear(intprod(curshape), outsize)
        self.outsize = outsize
        self.final_relu = final_relu

    def forward(self, x):
        x = x.to(dtype=th.float32) / self.scale_ob

        x = x.permute(0, 3, 1, 2)
        x = sequential(self.stacks, x, diag_name=self.name)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1)
        x = th.relu(x)
        x = self.dense(x)
        if self.final_relu:
            x = th.relu(x)
        return x

