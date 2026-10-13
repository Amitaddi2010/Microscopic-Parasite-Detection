import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2ParasiteDetector(nn.Module):
    """
    Self-Supervised Microscopy Foundation Model (DINOv2 ViT Backbone)
    with Multiscale Feature Pyramid and Dense Prediction Head for Helminth Egg Detection.
    """
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        # Load self-supervised DINOv2 Vision Transformer
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=pretrained)
        embed_dim = self.backbone.embed_dim # 384 for ViT-S/14
        
        # Feature adaptation: ViT patch embeddings to spatial feature maps
        self.proj_p3 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.proj_p4 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.proj_p5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Detection Head (Classification + BBox Regression)
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(256, 4, kernel_size=3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # Input must be divisible by patch size 14
        patch_h, patch_w = H // 14, W // 14
        
        # Extract patch tokens from DINOv2
        features = self.backbone.get_intermediate_layers(x, n=1)[0] # (B, num_patches, embed_dim)
        spatial_feats = features.transpose(1, 2).reshape(B, -1, patch_h, patch_w)
        
        # Multiscale feature generation
        p3 = self.proj_p3(spatial_feats)
        p4 = self.proj_p4(spatial_feats)
        p5 = self.proj_p5(p4)
        
        # Predictions on P3 scale
        cls_preds = self.cls_head(p3)
        reg_preds = self.reg_head(p3)
        
        return {
            "cls": cls_preds,
            "reg": reg_preds,
            "multiscale": [p3, p4, p5],
            "raw_feats": spatial_feats
        }
