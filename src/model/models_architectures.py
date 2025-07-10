import torch
import torch.nn as nn
import torch.nn.functional as F


class STMModel(nn.Module):
    def __init__(self, input_channels=1, seq_len=100):
        super(STMModel, self).__init__()
        self.seq_len = seq_len

        # --- Spatial features with small kernel ---
        self.spatial_small = nn.Sequential(
            nn.Conv1d(input_channels, 20, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(20, 30, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=5, stride=1, padding='same'),
            nn.ReLU()
        )

        # --- Spatial features with large kernel ---
        self.spatial_large = nn.Sequential(
            nn.Conv1d(input_channels, 20, kernel_size=10, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(20, 30, kernel_size=10, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=10, stride=1, padding='same'),
            nn.ReLU()
        )

        # --- Temporal features with BiGRU ---
        self.bigru1 = nn.GRU(input_size=1, hidden_size=16, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(input_size=32, hidden_size=32, batch_first=True, bidirectional=True)
        self.bigru3 = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)

        # --- CBAM after concatenation ---
        self.cbam = CBAM(in_channels=30 + 30 + 128)  # 30 (small) + 30 (large) + 128 (BiGRU last layer)

        # --- Output Module ---
        self.fc1 = nn.Linear((30 + 30 + 128) * seq_len, 1024)  # 188 * seq_len
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)
        self.output_linear = nn.Linear(1024, 1)
        self.output_sigmoid = nn.Linear(1024, 1)

    def forward(self, x):
        batch_size, _, seq_len = x.shape

        # Spatial features
        spatial_small = self.spatial_small(x)
        spatial_large = self.spatial_large(x)

        # Temporal features
        x_temp = x.permute(0, 2, 1)  # (B, T, C)
        out1, _ = self.bigru1(x_temp)
        out2, _ = self.bigru2(out1)
        out3, _ = self.bigru3(out2)
        temporal = out3.permute(0, 2, 1)  # (B, C, T)

        # Concatenate all features
        features = torch.cat([spatial_small, spatial_large, temporal], dim=1)

        # Attention mechanism
        Ft = self.cbam(features)  # (B, 188, T)

        # Output module
        flat = torch.flatten(Ft, start_dim=1)  # (B, 188 * T)
        fc = self.relu(self.fc1(flat))  # (B, 1024)

        linear_out = self.dropout(self.output_linear(fc))           # (B, 1)
        sigmoid_out = self.dropout(torch.sigmoid(self.output_sigmoid(fc)))  # (B, 1)

        output = linear_out * sigmoid_out  # element-wise multiplication

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)  # shape: (B, C, 1)
        return x * attention  # Multiply by channel attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)  # shape: (B, 2, T)
        attention = self.sigmoid(self.conv(concat))  # shape: (B, 1, T)
        return x * attention  # Multiply by channel attention


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class Seq2PointOneToOne(nn.Module):
    def __init__(self, input_length=599):
        super(Seq2PointOneToOne, self).__init__()

        # CNN layers — padding manually set to keep "same" output length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=30, kernel_size=11, padding=5)
        self.conv2 = nn.Conv1d(in_channels=30, out_channels=20, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(in_channels=20, out_channels=40, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(in_channels=40, out_channels=5,  kernel_size=5, padding=2)

        # Compute flattened size after convs
        self.flatten_input_size = self._get_flatten_size(input_length)

        # Dense layers
        self.fc1 = nn.Linear(self.flatten_input_size, 64)
        self.output = nn.Linear(64, 1)

    def _get_flatten_size(self, input_length):
        """Compute flattened size dynamically using dummy input."""
        dummy = torch.zeros(1, 1, input_length)
        x = self.conv1(dummy)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.view(1, -1).shape[1]  # flatten and get size

    def forward(self, x):
        """
        Forward pass
        Input shape: (batch_size, sequence_length, 1)
        """
        x = x.permute(0, 2, 1)  # to (batch_size, channels=1, sequence_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.output(x).squeeze()  # shape: (batch_size,)


MODEL_ARCHITECTURES = {
    'STMModel': STMModel,
    'Seq2PointOneToOne': Seq2PointOneToOne
}
