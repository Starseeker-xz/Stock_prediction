import torch
import torch.nn as nn

class LongTermBiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # out shape: (batch_size, seq_len, hidden_size * 2)
        out, (hn, cn) = self.lstm(x)
        return out, (hn, cn)


class MediumTermLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # out shape: (batch_size, seq_len, hidden_size)
        out, (hn, cn) = self.lstm(x)
        return out, (hn, cn)


class Chomp1d(nn.Module):
    """
    用于裁剪卷积后的填充，保证因果性 (Causal Convolution)
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN 的基本构建块: Dilated Causal Convolution + Residual Connection
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # 第一次卷积
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding) # 裁剪掉右侧多余的padding
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二次卷积
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # 残差连接的 1x1 卷积 (如果输入输出通道数不同)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class ShortTermTCNEncoder(nn.Module):
    """
    短期编码器 (Short-term Encoder)
    适用场景: 短期突发特征捕捉 (如: 1周数据)
    架构: TCN (Temporal Convolutional Network)
    """
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            input_size: 输入特征维度
            num_channels: 列表，表示每一层的通道数 (e.g. [32, 32, 64])
            kernel_size: 卷积核大小
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # TCN 需要输入 (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        y = self.network(x)
        # 转回 (batch_size, seq_len, output_size)
        return y.transpose(1, 2)
