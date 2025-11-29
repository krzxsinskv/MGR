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

        # NOTE: we DO NOT create ModuleList with modules already registered as attributes.
        # Creating ModuleList([...self.spatial_small...]) would re-register those submodules and may cause weird cycles.

        # --- CBAM after concatenation ---
        # pass weak ref via CBAM constructor (CBAM stores weakref internally)
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
                # defensive: in case input[0] not a tensor or requires grad
                pass

        # Register pre-forward hook on cbam itself
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

        linear_out = self.dropout(self.output_linear(fc))           # (B, 1)
        sigmoid_out = self.dropout(torch.sigmoid(self.output_sigmoid(fc)))  # (B, 1)

        output = linear_out * sigmoid_out

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, parent=None):
        super(ChannelAttention, self).__init__()
        self.parent = parent
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        reduced = max(1, in_channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_channels, reduced, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(reduced, in_channels, kernel_size=1, stride=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)  # shape: (B, C, 1)

        # use weakref stored on parent (CBAM) to update captured map
        if self.parent is not None and hasattr(self.parent, "model_ref") and self.parent.model_ref is not None:
            model_obj = self.parent.model_ref()
            if model_obj is not None:
                try:
                    model_obj.captured["channel_attention_map"] = attention[0].detach().cpu()
                except Exception:
                    pass

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

        if self.parent is not None and hasattr(self.parent, "model_ref") and self.parent.model_ref is not None:
            model_obj = self.parent.model_ref()
            if model_obj is not None:
                try:
                    model_obj.captured["spatial_attention_map"] = attention[0].detach().cpu()
                except Exception:
                    pass

        return x * attention


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, spatial_kernel_size=7, model=None):
        super(CBAM, self).__init__()
        # store only a weak reference to outer model to avoid PyTorch registering it as a Module
        self.model_ref = weakref.ref(model) if model is not None else None

        self.channel_attention = ChannelAttention(in_channels, reduction_ratio, parent=self)
        self.spatial_attention = SpatialAttention(spatial_kernel_size, parent=self)

    def forward(self, x):
        x_after_channel = self.channel_attention(x)

        # update captured if possible
        if self.model_ref is not None:
            model_obj = self.model_ref()
            if model_obj is not None:
                try:
                    model_obj.captured["post_channel"] = x_after_channel[0].detach().cpu()
                except Exception:
                    pass

        x_after_spatial = self.spatial_attention(x_after_channel)

        if self.model_ref is not None:
            model_obj = self.model_ref()
            if model_obj is not None:
                try:
                    model_obj.captured["post_spatial"] = x_after_spatial[0].detach().cpu()
                except Exception:
                    pass

        return x_after_spatial


MODEL_ARCHITECTURES = {
    'STMModel': STMModel
}


MODEL_ARCHITECTURES = {
    'STMModel': STMModel
}


