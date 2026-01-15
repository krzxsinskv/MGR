import torch
import torch.nn as nn
import weakref
import torch.nn.functional as F


class STMModel(nn.Module):
    def __init__(self, input_channels=1, seq_len=100):
        super(STMModel, self).__init__()
        self.seq_len = seq_len

        # Storage for hook outputs
        self.captured = {
            "fusion_features": None,
            "channel_attention_map": None,
            "post_channel": None,
            "spatial_attention_map": None,
            "post_spatial": None
        }

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
        # Note: expecting input channels = 1 -> bigru1 input_size=1
        self.bigru1 = nn.GRU(input_size=1, hidden_size=16, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(input_size=32, hidden_size=32, batch_first=True, bidirectional=True)
        self.bigru3 = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)

        self.cbam = CBAM(in_channels=30 + 30 + 128, model=self)

        # --- Output Module ---
        self.fc1 = nn.Linear((30 + 30 + 128) * seq_len, 1024)  # (188 * seq_len)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.output_linear = nn.Linear(1024, 1)
        self.output_sigmoid = nn.Linear(1024, 1)

        # Register HOOKS
        self.register_hooks()

    def register_hooks(self):
        # Hook BEFORE CBAM (input to CBAM = fusion features)
        def fusion_pre_hook(module, input):
            # input[0] is features fed to CBAM
            try:
                self.captured["fusion_features"] = input[0].detach().cpu()
            except Exception:
                pass

        self.cbam.register_forward_pre_hook(fusion_pre_hook)

    def forward(self, x):
        batch_size, _, seq_len = x.shape

        # Spatial features
        spatial_small = self.spatial_small(x)   # (B, 30, T)
        spatial_large = self.spatial_large(x)   # (B, 30, T)

        # Temporal features (B, T, C) for GRU
        x_temp = x.permute(0, 2, 1)  # (B, T, C)
        out1, _ = self.bigru1(x_temp)   # out1: (B, T, 32)
        out2, _ = self.bigru2(out1)     # out2: (B, T, 64)
        out3, _ = self.bigru3(out2)     # out3: (B, T, 128)
        temporal = out3.permute(0, 2, 1)  # (B, 128, T)

        # Concatenate all features -> channels: 30 + 30 + 128 = 188
        features = torch.cat([spatial_small, spatial_large, temporal], dim=1)  # (B, 188, T)

        # Attention mechanism
        Ft = self.cbam(features)  # (B, 188, T)

        # Output module
        flat = torch.flatten(Ft, start_dim=1)  # (B, 188 * T)
        fc = self.relu(self.fc1(flat))  # (B, 1024)
        ###
        # fc = self.dropout(fc)
        #
        # linear_out = self.output_linear(fc)
        # sigmoid_out = torch.sigmoid(self.output_sigmoid(fc))

        ###

        linear_out = self.dropout(self.output_linear(fc))           # (B, 1)
        sigmoid_out = self.dropout(torch.sigmoid(self.output_sigmoid(fc)))  # (B, 1)
        output = linear_out * sigmoid_out

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        reduced = max(1, in_channels // reduction_ratio)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(reduced, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention, attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg, max_], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention, attention


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7, model=None):
        super().__init__()
        self.model_ref = weakref.ref(model) if model else None
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        model = self.model_ref() if self.model_ref else None

        x_ca, ca_map = self.channel_attention(x)
        if model:
            model.captured["channel_attention_map"] = ca_map.detach().cpu()
            model.captured["post_channel"] = x_ca.detach().cpu()

        x_sa, sa_map = self.spatial_attention(x_ca)
        if model:
            model.captured["spatial_attention_map"] = sa_map.detach().cpu()
            model.captured["post_spatial"] = x_sa.detach().cpu()

        return x_sa


MODEL_ARCHITECTURES = {
    'STMModel': STMModel
}
