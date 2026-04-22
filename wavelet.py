import torch
import torch.nn as nn

def dwt_init(x):
    # 行降采样
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2

    # 列降采样
    x1 = x01[:, :, :, 0::2]  # 左上
    x2 = x02[:, :, :, 0::2]  # 左下
    x3 = x01[:, :, :, 1::2]  # 右上
    x4 = x02[:, :, :, 1::2]  # 右下

    # 四个子带
    x_LL =  x1 +  x2 +  x3 +  x4
    x_HL = -x1 -  x2 +  x3 +  x4
    x_LH = -x1 +  x2 -  x3 +  x4
    x_HH =  x1 -  x2 -  x3 +  x4

    return x_LL, x_LH, x_HL, x_HH


def iwt_init(x_LL, x_LH, x_HL, x_HH):
    r = 2
    B, C, H, W = x_LL.size()
    out = torch.zeros((B, C, H*r, W*r), dtype=x_LL.dtype, device=x_LL.device)

    out[:, :, 0::2, 0::2] = x_LL - x_HL - x_LH + x_HH  # Corresponds to 4 * x1 from dwt_init
    out[:, :, 1::2, 0::2] = x_LL - x_HL + x_LH - x_HH  # Corresponds to 4 * x2 from dwt_init
    out[:, :, 0::2, 1::2] = x_LL + x_HL - x_LH - x_HH  # Corresponds to 4 * x3 from dwt_init
    out[:, :, 1::2, 1::2] = x_LL + x_HL + x_LH + x_HH  # Corresponds to 4 * x4 from dwt_init
    
    return out / 2 



class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x_LL, x_LH, x_HL, x_HH):
        return iwt_init(x_LL, x_LH, x_HL, x_HH)

