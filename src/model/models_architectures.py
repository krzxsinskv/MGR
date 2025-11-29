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
        self.bigru1 = nn.GRU(input_size=1, hidden_size=16, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(input_size=32, hidden_size=32, batch_first=True, bidirectional=True)
        self.bigru3 = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)

        # --- CBAM after concatenation ---
        self.cbam = CBAM(in_channels=30 + 30 + 128, model=self)  # 30 (small) + 30 (large) + 128 (BiGRU last layer)

        # --- Output Module ---
        self.fc1 = nn.Linear((30 + 30 + 128) * seq_len, 1024)  # 188 * seq_len
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)
        self.output_linear = nn.Linear(1024, 1)
        self.output_sigmoid = nn.Linear(1024, 1)

        # Register HOOKS
        self.register_hooks()

    def register_hooks(self):
        # Hook BEFORE CBAM (input to CBAM = fusion features)
        def fusion_pre_hook(module, input):
            self.captured["fusion_features"] = input[0].detach().cpu()

        self.cbam.register_forward_pre_hook(fusion_pre_hook)

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

        # LINEAR ACTIVATION
        # output = self.output_linear(fc)

        # ARTICLE'S ACTIVATION
        output = linear_out * sigmoid_out

        # RELU ACTIVATION
        # output = F.relu(self.output_linear(fc)) # element-wise multiplication

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, parent=None):
        super(ChannelAttention, self).__init__()
        self.parent = parent
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

        # Access parent's weakref to update captured map if available
        if self.parent is not None and hasattr(self.parent, "model_ref"):
            parent_model = self.parent.model_ref()
            if parent_model is not None:
                parent_model.captured["channel_attention_map"] = attention[0].detach().cpu()

        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, parent=None):
        super(SpatialAttention, self).__init__()
        self.parent = parent
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, T)
        attention = self.sigmoid(self.conv(concat))     # (B, 1, T)

        if self.parent is not None and hasattr(self.parent, "model_ref"):
            parent_model = self.parent.model_ref()
            if parent_model is not None:
                parent_model.captured["spatial_attention_map"] = attention[0].detach().cpu()

        return x * attention


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, spatial_kernel_size=7, model=None):
        super(CBAM, self).__init__()
        # store weak reference to outer model to avoid PyTorch registering it as submodule
        self.model_ref = weakref.ref(model) if model is not None else None

        self.channel_attention = ChannelAttention(in_channels, reduction_ratio, parent=self)
        self.spatial_attention = SpatialAttention(spatial_kernel_size, parent=self)

    def forward(self, x):
        x_after_channel = self.channel_attention(x)

        # Update captured via weakref if possible
        if self.model_ref is not None:
            parent_model = self.model_ref()
            if parent_model is not None:
                parent_model.captured["post_channel"] = x_after_channel[0].detach().cpu()

        x_after_spatial = self.spatial_attention(x_after_channel)

        if self.model_ref is not None:
            parent_model = self.model_ref()
            if parent_model is not None:
                parent_model.captured["post_spatial"] = x_after_spatial[0].detach().cpu()

        return x_after_spatial


MODEL_ARCHITECTURES = {
    'STMModel': STMModel
}


