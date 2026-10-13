import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSMBlock(nn.Module):
    """
    Selective State Space Model (SSM) / Visual Mamba Block.
    Performs bidirectional spatial state-space scanning with linear O(N) complexity,
    enabling global microscopic receptive fields without quadratic self-attention memory overhead.
    """
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand)
        self.d_state = d_state
        
        # Linear projection to expanded state space
        self.in_proj = nn.Conv2d(d_model, self.d_inner * 2, kernel_size=1, bias=False)
        self.act = nn.SiLU()
        
        # Depthwise 1D/2D convolution for localized spatial continuity
        self.dw_conv = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner, bias=True)
        
        # State parameter matrices (discretized continuous SSM)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Conv2d(self.d_inner, d_model, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, x):
        res = x
        B, C, H, W = x.shape
        L = H * W
        
        proj = self.in_proj(x)
        x_branch, gate = proj.chunk(2, dim=1)
        
        x_conv = self.act(self.dw_conv(x_branch))
        
        # Flatten spatial dimensions for sequence state-space scanning
        # Shape: (B, d_inner, L) -> (B, L, d_inner)
        u = x_conv.flatten(2).transpose(1, 2)
        
        # Selective scanning approximation (bidirectional recurrence)
        delta = F.softplus(u)
        B_val = self.B_proj(u) # (B, L, d_state)
        C_val = self.C_proj(u) # (B, L, d_state)
        
        # Global context vector via state memory integration
        state = torch.einsum('bld,bls->bds', u, B_val) / (L ** 0.5)
        y = torch.einsum('bds,bls->bld', state, C_val) + u * self.D
        
        y_2d = y.transpose(1, 2).view(B, self.d_inner, H, W)
        y_gated = y_2d * torch.sigmoid(gate)
        
        out = self.norm(self.out_proj(y_gated)) + res
        return out


class MLCABlock(nn.Module):
    """
    Multi-Level Channel Attention (MLCA) block from YAC-Net.
    Extracts multi-scale channel statistics across local and global contextual pools
    to highlight translucent parasite shells against noisy fecal debris.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        mid_channels = max(channels // reduction, 8)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveAvgPool2d(3)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        # Global attention
        g_att = self.mlp(self.global_pool(x))
        # Local contextual attention
        l_att = self.mlp(self.local_pool(x))
        l_att = F.interpolate(l_att, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        att = self.sigmoid(g_att + l_att)
        return res * att


class AFPNNeck(nn.Module):
    """
    Asymptotic Feature Pyramid Network (AFPN) Neck.
    Smoothly fuses adjacent and non-adjacent feature hierarchies to avoid small-scale feature dilution.
    """
    def __init__(self, in_channels_list=[128, 256, 512], out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU()
            ) for _ in in_channels_list
        ])
        self.mlca_blocks = nn.ModuleList([
            MLCABlock(out_channels) for _ in in_channels_list
        ])

    def forward(self, features):
        # features: [P3, P4, P5]
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Asymptotic progressive top-down and bottom-up fusion
        # P5 -> P4
        p5_up = F.interpolate(laterals[2], size=laterals[1].shape[2:], mode='nearest')
        laterals[1] = laterals[1] + p5_up
        # P4 -> P3
        p4_up = F.interpolate(laterals[1], size=laterals[0].shape[2:], mode='nearest')
        laterals[0] = laterals[0] + p4_up
        
        # Bottom-up refinement
        # P3 -> P4
        p3_down = F.adaptive_avg_pool2d(laterals[0], output_size=laterals[1].shape[2:])
        laterals[1] = laterals[1] + p3_down
        # P4 -> P5
        p4_down = F.adaptive_avg_pool2d(laterals[1], output_size=laterals[2].shape[2:])
        laterals[2] = laterals[2] + p4_down
        
        outs = []
        for l, smooth, mlca in zip(laterals, self.smooth_convs, self.mlca_blocks):
            out = smooth(l)
            out = mlca(out)
            outs.append(out)
        return outs
