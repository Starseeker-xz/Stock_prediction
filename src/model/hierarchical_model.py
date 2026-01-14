import torch
import torch.nn as nn
from src.config import ModelConfig
from src.model.encoders import LongTermBiLSTMEncoder, MediumTermLSTMEncoder, ShortTermTCNEncoder
from src.model.attention import CrossAttention

class HierarchicalStockModel(nn.Module):
    def __init__(self):
        super(HierarchicalStockModel, self).__init__()
        
        config = ModelConfig
        self.mode = getattr(config, 'MODEL_MODE', 'full')
        print(f"Initializing model in mode: {self.mode}")
        
        # 1. Encoders & Projections
        if self.mode in ['full', 'long_only']:
            self.long_encoder = LongTermBiLSTMEncoder(
                input_size=config.INPUT_SIZE,
                hidden_size=config.HIDDEN_SIZE_LONG,
                dropout=config.DROPOUT
            )
            # BiLSTM output is hidden_size * 2
            self.proj_long = nn.Linear(config.HIDDEN_SIZE_LONG * 2, config.D_MODEL)
            
        if self.mode in ['full', 'medium_only']:
            self.medium_encoder = MediumTermLSTMEncoder(
                input_size=config.INPUT_SIZE,
                hidden_size=config.HIDDEN_SIZE_MEDIUM,
                dropout=config.DROPOUT
            )
            # LSTM output is hidden_size
            self.proj_medium = nn.Linear(config.HIDDEN_SIZE_MEDIUM, config.D_MODEL)
            
        if self.mode in ['full', 'short_only']:
            self.short_encoder = ShortTermTCNEncoder(
                input_size=config.INPUT_SIZE,
                num_channels=config.TCN_CHANNELS,
                dropout=config.DROPOUT
            )
            # TCN output is last channel size
            self.proj_short = nn.Linear(config.TCN_CHANNELS[-1], config.D_MODEL)
        
        # 3. Cross Attention Layers (Only for Full Mode)
        if self.mode == 'full':
            # Layer 1: Medium (Query) attends to Long (Key/Value)
            self.cross_attn_1 = CrossAttention(
                d_model=config.D_MODEL,
                n_heads=config.NUM_HEADS,
                dropout=config.DROPOUT
            )
            
            # Layer 2: Short (Query) attends to Enriched Medium (Key/Value)
            self.cross_attn_2 = CrossAttention(
                d_model=config.D_MODEL,
                n_heads=config.NUM_HEADS,
                dropout=config.DROPOUT
            )
        
        # 4. Output Head (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(config.D_MODEL, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, config.OUTPUT_SIZE)
        )

    def forward(self, x_long, x_medium, x_short):
        """
        Args:
            x_long: (batch_size, seq_len_long, input_size)
            x_medium: (batch_size, seq_len_medium, input_size)
            x_short: (batch_size, seq_len_short, input_size)
        """
        
        if self.mode == 'full':
            # --- Encoding ---
            long_feat, _ = self.long_encoder(x_long)
            medium_feat, _ = self.medium_encoder(x_medium)
            short_feat = self.short_encoder(x_short)
            
            # --- Projection ---
            long_proj = self.proj_long(long_feat)
            medium_proj = self.proj_medium(medium_feat)
            short_proj = self.proj_short(short_feat)
            
            # --- Hierarchical Mixing ---
            
            # 1. Medium queries Long
            medium_enriched, _ = self.cross_attn_1(
                query=medium_proj,
                key=long_proj,
                value=long_proj
            )
            
            # 2. Short queries Enriched Medium
            short_enriched, _ = self.cross_attn_2(
                query=short_proj,
                key=medium_enriched,
                value=medium_enriched
            )
            
            # --- Prediction ---
            # Take the last time step of the short term sequence
            last_step_feat = short_enriched[:, -1, :]
            return self.mlp(last_step_feat)
            
        elif self.mode == 'long_only':
            long_feat, _ = self.long_encoder(x_long)
            long_proj = self.proj_long(long_feat)
            # Take last time step
            return self.mlp(long_proj[:, -1, :])

        elif self.mode == 'medium_only':
            medium_feat, _ = self.medium_encoder(x_medium)
            medium_proj = self.proj_medium(medium_feat)
            return self.mlp(medium_proj[:, -1, :])
            
        elif self.mode == 'short_only':
            short_feat = self.short_encoder(x_short)
            short_proj = self.proj_short(short_feat)
            return self.mlp(short_proj[:, -1, :])
        
        else:
             raise ValueError(f"Unknown mode: {self.mode}")
