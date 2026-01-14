import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    """
    Cross Attention Layer
    用于不同序列或特征之间的交互混合。
    例如: 用短期特征(Query)去查询长期特征(Key/Value)，从而融合长期趋势信息。
    """
    def __init__(self, d_model, n_heads, dropout=0.1, dim_feedforward=None):
        super(CrossAttention, self).__init__()
        
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
            
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        
        # Add & Norm 层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Args:
            query: (batch_size, query_len, d_model) - 通常是主要关注的序列 (如短期特征)
            key:   (batch_size, key_len, d_model)   - 提供上下文的序列 (如长期特征)
            value: (batch_size, value_len, d_model) - 通常与 key 相同
            key_padding_mask: (batch_size, key_len) - 掩码，True 表示忽略该位置
            attn_mask: (query_len, key_len) - 注意力掩码
            
        Returns:
            output: (batch_size, query_len, d_model) - 混合后的特征
            attn_weights: (batch_size, num_heads, query_len, key_len) - 注意力权重
        """
        # 1. Cross Attention
        # attn_output shape: (batch_size, query_len, d_model)
        attn_output, attn_weights = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        # 2. Add & Norm (Residual Connection)
        x = self.norm1(query + self.dropout(attn_output))
        
        # 3. Feed Forward Network
        ff_output = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        
        # 4. Add & Norm (Residual Connection)
        output = self.norm2(x + self.dropout2(ff_output))
        
        return output, attn_weights
