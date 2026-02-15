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
        self.gate_detach_aux = bool(getattr(config, 'GATE_DETACH_AUX', False))

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

        # Optional branch normalization before fusion
        # NOTE:
        # For Close-level forecasting, short branch amplitude carries strong level info.
        # Normalizing short branch may hurt extrapolation and lead to near-constant outputs.
        self.branch_layer_norm = bool(getattr(config, 'BRANCH_LAYER_NORM', True))
        self.branch_layer_norm_mode = str(getattr(config, 'BRANCH_LAYER_NORM_MODE', 'aux_only'))
        # mode: 'all' | 'aux_only' | 'none'
        self.norm_short_enabled = self.branch_layer_norm and self.branch_layer_norm_mode == 'all'
        self.norm_aux_enabled = self.branch_layer_norm and self.branch_layer_norm_mode in ('all', 'aux_only')
        if self.branch_layer_norm:
            if self.norm_short_enabled:
                self.norm_short = nn.LayerNorm(config.D_MODEL)
            if self.use_medium and self.norm_aux_enabled:
                self.norm_medium = nn.LayerNorm(config.D_MODEL)
            if self.use_long and self.norm_aux_enabled:
                self.norm_long = nn.LayerNorm(config.D_MODEL)

        if self.fusion_method == 'concat':
            fusion_dim = config.D_MODEL * (1 + int(self.use_medium) + int(self.use_long))
        elif self.fusion_method == 'gated_short_mlp':
            fusion_dim = config.D_MODEL
            gate_out_dim = int(self.use_medium) + int(self.use_long)
            if gate_out_dim > 0:
                gate_in_dim = config.D_MODEL * (1 + gate_out_dim)
                gate_hidden = int(getattr(config, 'GATE_HIDDEN_SIZE', 32))
                self.gate_norm_short = nn.LayerNorm(config.D_MODEL)
                if self.use_medium:
                    self.gate_norm_medium = nn.LayerNorm(config.D_MODEL)
                if self.use_long:
                    self.gate_norm_long = nn.LayerNorm(config.D_MODEL)
                self.gate_mlp = nn.Sequential(
                    nn.Linear(gate_in_dim, gate_hidden),
                    nn.ReLU(),
                    nn.Linear(gate_hidden, gate_out_dim),
                )
                gate_bias = float(getattr(config, 'GATE_INIT_BIAS', -2.0))
                nn.init.constant_(self.gate_mlp[-1].bias, gate_bias)
        elif self.fusion_method == 'softmax_short_mlp':
            # Use short feature to produce source weights via softmax over
            # [short, medium?, long?], then weighted sum to D_MODEL.
            fusion_dim = config.D_MODEL
            source_count = 1 + int(self.use_medium) + int(self.use_long)
            gate_in_dim = config.D_MODEL * source_count
            gate_hidden = int(getattr(config, 'GATE_HIDDEN_SIZE', 32))
            self.gate_norm_short = nn.LayerNorm(config.D_MODEL)
            if self.use_medium:
                self.gate_norm_medium = nn.LayerNorm(config.D_MODEL)
            if self.use_long:
                self.gate_norm_long = nn.LayerNorm(config.D_MODEL)
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_in_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, source_count),
            )
            gate_bias = float(getattr(config, 'GATE_INIT_BIAS', -2.0))
            nn.init.constant_(self.gate_mlp[-1].bias, gate_bias)
        else:
            raise ValueError(
                f"Unknown FUSION_METHOD: {self.fusion_method}. "
                "Expected one of: 'concat', 'gated_short_mlp', 'softmax_short_mlp'."
            )

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, config.OUTPUT_SIZE),
        )
        self._last_gates = None

        # Instance Normalization (RevIN-style)
        # 对每个样本的输入窗口独立归一化，消除不同时间段的价格水平漂移
        # 输出通过反归一化映射回全局标准化空间
        self.instance_norm = bool(getattr(config, 'INSTANCE_NORM', False))
        if self.instance_norm:
            target_col = getattr(config, 'TARGET_COL', 'Close')
            feature_cols_sm = list(getattr(config, 'FEATURE_COLS_SHORT_MEDIUM', config.FEATURE_COLS))
            self.close_idx_sm = feature_cols_sm.index(
                target_col if target_col in feature_cols_sm else 'Close'
            )
            print(f"  Instance Normalization enabled (denorm via feature idx={self.close_idx_sm})")

    def _instance_normalize(self, x, eps=1e-5):
        """RevIN-style per-instance normalization over the time dimension.

        Args:
            x: (B, T, C) input tensor
        Returns:
            x_norm: (B, T, C)  normalized tensor
            mean:   (B, 1, C)  per-instance per-feature mean
            std:    (B, 1, C)  per-instance per-feature std (clamped)
        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=eps)
        return (x - mean) / std, mean, std

    def forward(self, x_long, x_medium, x_short):
        """Forward.

        Args:
            x_long: (batch_size, seq_len_long, input_size) or None
            x_medium: (batch_size, seq_len_medium, input_size) or None
            x_short: (batch_size, seq_len_short, input_size)
        """

        if x_short is None:
            raise ValueError("x_short must not be None (short-term backbone is required).")

        # --- Instance Normalization (RevIN) ---
        if self.instance_norm:
            x_short, _s_mean, _s_std = self._instance_normalize(x_short)
            # 保留 target 列（Close）的统计量，用于输出反归一化
            denorm_mean = _s_mean[:, 0, self.close_idx_sm]  # (B,)
            denorm_std = _s_std[:, 0, self.close_idx_sm]     # (B,)
            if self.use_medium and x_medium is not None:
                x_medium, _, _ = self._instance_normalize(x_medium)
            if self.use_long and x_long is not None:
                x_long, _, _ = self._instance_normalize(x_long)

        # --- Short backbone ---
        short_feat = self.short_encoder(x_short)
        short_proj = self.proj_short(short_feat)
        short_last = short_proj[:, -1, :]
        if self.norm_short_enabled:
            short_last = self.norm_short(short_last)

        if self.fusion_method == 'concat':
            fused_parts = [short_last]
        else:
            fused = short_last
            gates = []
            gate_input_parts = [self.gate_norm_short(short_last)]

        # --- Optional medium ---
        if self.use_medium:
            if x_medium is None:
                raise ValueError("x_medium is required when MODEL_MODE uses medium.")
            medium_feat, _ = self.medium_encoder(x_medium)
            medium_proj = self.proj_medium(medium_feat)
            medium_last = medium_proj[:, -1, :]
            if self.norm_aux_enabled:
                medium_last = self.norm_medium(medium_last)
            if self.fusion_method == 'concat':
                fused_parts.append(medium_last)
            else:
                gates.append(medium_last)
                gate_input_parts.append(self.gate_norm_medium(medium_last))

        # --- Optional long ---
        if self.use_long:
            if x_long is None:
                raise ValueError("x_long is required when MODEL_MODE uses long.")
            long_feat, _ = self.long_encoder(x_long)
            long_proj = self.proj_long(long_feat)
            long_last = long_proj[:, -1, :]
            if self.norm_aux_enabled:
                long_last = self.norm_long(long_last)
            if self.fusion_method == 'concat':
                fused_parts.append(long_last)
            else:
                gates.append(long_last)
                gate_input_parts.append(self.gate_norm_long(long_last))

        # --- Fusion → output ---
        if self.fusion_method == 'concat':
            output = self.mlp(torch.cat(fused_parts, dim=-1))
        elif len(gates) == 0:
            self._last_gates = None
            output = self.mlp(fused)
        else:
            gate_input = torch.cat(gate_input_parts, dim=-1)
            gate_logits = self.gate_mlp(gate_input)
            if self.fusion_method == 'gated_short_mlp':
                gate_weights = torch.sigmoid(gate_logits)  # (B, n_aux)
                for idx, aux_last in enumerate(gates):
                    aux = aux_last.detach() if self.gate_detach_aux else aux_last
                    w = gate_weights[:, idx].unsqueeze(-1)  # (B, 1)
                    fused = fused + w * aux

                self._last_gates = gate_weights.detach()
                output = self.mlp(fused)
            else:
                # --- softmax_short_mlp ---
                source_weights = torch.softmax(gate_logits, dim=-1)  # (B, 1 + n_aux)
                fused = source_weights[:, 0].unsqueeze(-1) * short_last
                for idx, aux_last in enumerate(gates):
                    w = source_weights[:, idx + 1].unsqueeze(-1)
                    fused = fused + w * aux_last

                # Keep backward-compatible: _last_gates stores aux branch weights only.
                self._last_gates = source_weights[:, 1:].detach()
                output = self.mlp(fused)

        # --- Instance Denormalization (RevIN reverse) ---
        if self.instance_norm:
            output = output * denorm_std.unsqueeze(-1) + denorm_mean.unsqueeze(-1)

        return output