import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualJevModel(nn.Module):
    """
    Visual Jev (arXiv:2609.25845, Sept 2026 - CMU).
    Shared Visual Context Architecture for Fast Multi-Attribute Diagnostic Decisions:
    1. Encodes the high-resolution microscopic image ONCE via a vision backbone.
    2. Concurrently evaluates multiple discrete diagnostic questions:
       - Query 1: Helminth Parasite Presence (Binary: Negative vs Positive)
       - Query 2: Species Identification (8-Class Categorical Distribution)
       - Query 3: Multi-Infection Indicator (Single vs Multiple Ova)
       - Query 4: Debris/Artifact Severity Filter (Clean vs Heavy Fecal Detritus)
    3. Reads candidate decision logits directly from parallel discriminative readout heads
       without autoregressive text generation or hallucination risk.
    """
    def __init__(self, num_classes=8, embed_dim=512):
        super().__init__()
        # Shared Context Vision Backbone (Efficient ResNet/ConvNeXt feature extractor)
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        backbone = convnext_tiny(weights=weights)
        self.encoder = backbone.features
        in_features = backbone.classifier[2].in_features
        
        self.context_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.context_proj = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Shared Suffix Query Embeddings (Visual Jev prompt conditioning)
        self.query_presence = nn.Parameter(torch.randn(1, embed_dim))
        self.query_species = nn.Parameter(torch.randn(1, embed_dim))
        self.query_density = nn.Parameter(torch.randn(1, embed_dim))
        self.query_artifact = nn.Parameter(torch.randn(1, embed_dim))
        
        # Discriminative Readout Heads (Direct Decision Logits)
        self.head_presence = nn.Linear(embed_dim, 2)
        self.head_species = nn.Linear(embed_dim, num_classes)
        self.head_density = nn.Linear(embed_dim, 3) # 0, 1, multiple
        self.head_artifact = nn.Linear(embed_dim, 2) # low vs high debris

    def forward(self, x):
        # 1. Single forward pass shared visual context
        feats = self.encoder(x)
        ctx = self.context_pool(feats).flatten(1)
        shared_emb = self.context_proj(ctx) # (B, embed_dim)
        
        # 2. Parallel query suffix evaluation (Batched discriminative readout)
        presence_logits = self.head_presence(shared_emb + self.query_presence)
        species_logits = self.head_species(shared_emb + self.query_species)
        density_logits = self.head_density(shared_emb + self.query_density)
        artifact_logits = self.head_artifact(shared_emb + self.query_artifact)
        
        return {
            "presence": presence_logits,
            "species": species_logits,
            "density": density_logits,
            "artifact": artifact_logits,
            "features": feats
        }
