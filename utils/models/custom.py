import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv, Bottleneck

class VisualTransformerLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super(VisualTransformerLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        # Reshape x into a sequence of vectors
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)  # [batch_size, num_pixels, d_model]
        
        # Linear projections to obtain Q, K, V tensors
        q = self.query(x)  # [batch_size, num_pixels, d_model]
        k = self.key(x)  # [batch_size, num_pixels, d_model]
        v = self.value(x)  # [batch_size, num_pixels, d_model]
        
        # Compute attention scores and weighted sum of values
        attn_output, _ = self.attention(q, k, v)  # [batch_size, num_pixels, d_model]
        attn_output = self.dropout(attn_output)
        x = self.norm(x + attn_output)
        
        # Reshape output back into the original image shape
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))