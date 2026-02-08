import torch
import torch.nn as nn
from src.config import ModelConfig
from src.model.encoders import LongTermBiLSTMEncoder, MediumTermLSTMEncoder, ShortTermTCNEncoder
class HierarchicalStockModel(nn.Module):
    """Hierarchical model with short-term module as the backbone.

    Supported modes (MODEL_MODE):
    - 'short_only'   : short-term only
    - 'short_medium' : short + medium
    - 'short_long'   : short + long
    - 'full'         : short + medium + long
    """

    def __init__(self):
        super().__init__()

        config = ModelConfig
        self.mode = getattr(config, 'MODEL_MODE', 'full')
        self.fusion_method = getattr(config, 'FUSION_METHOD', 'concat')

        if self.mode == 'short_only':
            self.use_short, self.use_medium, self.use_long = True, False, False
        elif self.mode == 'short_medium':
            self.use_short, self.use_medium, self.use_long = True, True, False
        elif self.mode == 'short_long':
            self.use_short, self.use_medium, self.use_long = True, False, True
        elif self.mode == 'full':
            self.use_short, self.use_medium, self.use_long = True, True, True
        else:
            raise ValueError(
                f"Unknown MODEL_MODE: {self.mode}. "
                "Expected one of: 'short_only', 'short_medium', 'short_long', 'full'."
            )

        print(f"Initializing model in mode: {self.mode}")

        # Short-term backbone is always enabled
        self.short_encoder = ShortTermTCNEncoder(
            input_size=config.INPUT_SIZE_SHORT_MEDIUM,
            num_channels=config.TCN_CHANNELS,
            dropout=config.DROPOUT,
        )
        self.proj_short = nn.Linear(config.TCN_CHANNELS[-1], config.D_MODEL)

        if self.use_medium:
            self.medium_encoder = MediumTermLSTMEncoder(
                input_size=config.INPUT_SIZE_SHORT_MEDIUM,
                hidden_size=config.HIDDEN_SIZE_MEDIUM,
                dropout=config.DROPOUT,
            )
            self.proj_medium = nn.Linear(config.HIDDEN_SIZE_MEDIUM, config.D_MODEL)

        if self.use_long:
            self.long_encoder = LongTermBiLSTMEncoder(
                input_size=config.INPUT_SIZE_LONG,
                hidden_size=config.HIDDEN_SIZE_LONG,
                dropout=config.DROPOUT,
            )
            # BiLSTM output is hidden_size * 2
            self.proj_long = nn.Linear(config.HIDDEN_SIZE_LONG * 2, config.D_MODEL)

        if self.fusion_method == 'concat':
            fusion_dim = config.D_MODEL * (1 + int(self.use_medium) + int(self.use_long))
        elif self.fusion_method == 'gated_short_mlp':
            fusion_dim = config.D_MODEL
            gate_out_dim = int(self.use_medium) + int(self.use_long)
            if gate_out_dim > 0:
                gate_hidden = int(getattr(config, 'GATE_HIDDEN_SIZE', 32))
                self.gate_mlp = nn.Sequential(
                    nn.Linear(config.D_MODEL, gate_hidden),
                    nn.ReLU(),
                    nn.Linear(gate_hidden, gate_out_dim),
                )
                gate_bias = float(getattr(config, 'GATE_INIT_BIAS', -2.0))
                nn.init.constant_(self.gate_mlp[-1].bias, gate_bias)
        else:
            raise ValueError(
                f"Unknown FUSION_METHOD: {self.fusion_method}. "
                "Expected one of: 'concat', 'gated_short_mlp'."
            )

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, config.OUTPUT_SIZE),
        )
        self._last_gates = None

    def forward(self, x_long, x_medium, x_short):
        """Forward.

        Args:
            x_long: (batch_size, seq_len_long, input_size) or None
            x_medium: (batch_size, seq_len_medium, input_size) or None
            x_short: (batch_size, seq_len_short, input_size)
        """

        if x_short is None:
            raise ValueError("x_short must not be None (short-term backbone is required).")

        # --- Short backbone ---
        short_feat = self.short_encoder(x_short)
        short_proj = self.proj_short(short_feat)
        short_last = short_proj[:, -1, :]

        if self.fusion_method == 'concat':
            fused_parts = [short_last]
        else:
            fused = short_last
            gates = []

        # --- Optional medium ---
        if self.use_medium:
            if x_medium is None:
                raise ValueError("x_medium is required when MODEL_MODE uses medium.")
            medium_feat, _ = self.medium_encoder(x_medium)
            medium_proj = self.proj_medium(medium_feat)
            medium_last = medium_proj[:, -1, :]
            if self.fusion_method == 'concat':
                fused_parts.append(medium_last)
            else:
                gates.append(medium_last)

        # --- Optional long ---
        if self.use_long:
            if x_long is None:
                raise ValueError("x_long is required when MODEL_MODE uses long.")
            long_feat, _ = self.long_encoder(x_long)
            long_proj = self.proj_long(long_feat)
            long_last = long_proj[:, -1, :]
            if self.fusion_method == 'concat':
                fused_parts.append(long_last)
            else:
                gates.append(long_last)

        if self.fusion_method == 'concat':
            fused_feat = torch.cat(fused_parts, dim=-1)
            return self.mlp(fused_feat)

        # --- gated_short_mlp ---
        if len(gates) == 0:
            self._last_gates = None
            return self.mlp(fused)

        gate_logits = self.gate_mlp(short_last)
        gate_weights = torch.sigmoid(gate_logits)  # (B, n_aux)

        gate_detach_aux = bool(getattr(ModelConfig, 'GATE_DETACH_AUX', False))
        for idx, aux_last in enumerate(gates):
            aux = aux_last.detach() if gate_detach_aux else aux_last
            w = gate_weights[:, idx].unsqueeze(-1)  # (B, 1)
            fused = fused + w * aux

        self._last_gates = gate_weights.detach()
        return self.mlp(fused)