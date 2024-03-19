import torch
from torch import nn


class GridGradientCentralDiff:
    def __init__(self, nc, padding=True, diagonal=False):
        self.conv_x = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_y = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_xy = None
        if diagonal:
            self.conv_xy = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)

        self.padding = None
        if padding:
            self.padding = nn.ReplicationPad2d([0, 1, 0, 1])

        fx = torch.zeros(nc, nc, 2, 2).float().cuda()
        fy = torch.zeros(nc, nc, 2, 2).float().cuda()
        if diagonal:
            fxy = torch.zeros(nc, nc, 2, 2).float().cuda()

        fx_ = torch.tensor([[1, -1], [0, 0]]).cuda()
        fy_ = torch.tensor([[1, 0], [-1, 0]]).cuda()
        if diagonal:
            fxy_ = torch.tensor([[1, 0], [0, -1]]).cuda()

        for i in range(nc):
            fx[i, i, :, :] = fx_
            fy[i, i, :, :] = fy_
            if diagonal:
                fxy[i, i, :, :] = fxy_

        self.conv_x.weight = nn.Parameter(fx)
        self.conv_y.weight = nn.Parameter(fy)
        if diagonal:
            self.conv_xy.weight = nn.Parameter(fxy)

    def __call__(self, grid_2d):
        _image = grid_2d
        if self.padding is not None:
            _image = self.padding(_image)
        dx = self.conv_x(_image)
        dy = self.conv_y(_image)

        if self.conv_xy is not None:
            dxy = self.conv_xy(_image)
            return dx, dy, dxy
        return dx, dy


class EdgeAwareVariationLoss(nn.Module):
    def __init__(self, in1_nc, grad_fn=GridGradientCentralDiff):
        super(EdgeAwareVariationLoss, self).__init__()
        self.in1_grad_fn = grad_fn(in1_nc)
        # self.in2_grad_fn = grad_fn(in2_nc)

    def forward(self, in1, mean=False):
        in1_dx, in1_dy = self.in1_grad_fn(in1)
        # in2_dx, in2_dy = self.in2_grad_fn(in2)

        abs_in1_dx, abs_in1_dy = in1_dx.abs().sum(dim=1, keepdim=True), in1_dy.abs().sum(dim=1, keepdim=True)
        # abs_in2_dx, abs_in2_dy = in2_dx.abs().sum(dim=1,keepdim=True), in2_dy.abs().sum(dim=1,keepdim=True)

        weight_dx, weight_dy = torch.exp(-abs_in1_dx), torch.exp(-abs_in1_dy)

        variation = weight_dx * abs_in1_dx + weight_dy * abs_in1_dy

        if mean != False:
            return variation.mean()
        return variation.sum()


class GrayEdgeAwareVariationLoss(nn.Module):
    def __init__(self, in1_nc, in2_nc, grad_fn=GridGradientCentralDiff):
        super(GrayEdgeAwareVariationLoss, self).__init__()
        self.in1_grad_fn = grad_fn(in1_nc)  # Gray
        self.in2_grad_fn = grad_fn(in2_nc)  # Sharp

    def forward(self, in1, in2, mean=False):
        in1_dx, in1_dy = self.in1_grad_fn(in1)
        in2_dx, in2_dy = self.in2_grad_fn(in2)

        abs_in1_dx, abs_in1_dy = in1_dx.abs().sum(dim=1, keepdim=True), in1_dy.abs().sum(dim=1, keepdim=True)
        abs_in2_dx, abs_in2_dy = in2_dx.abs().sum(dim=1, keepdim=True), in2_dy.abs().sum(dim=1, keepdim=True)

        weight_dx, weight_dy = torch.exp(-abs_in2_dx), torch.exp(-abs_in2_dy)

        variation = weight_dx * abs_in1_dx + weight_dy * abs_in1_dy

        if mean != False:
            return variation.mean()
        return variation.sum()
