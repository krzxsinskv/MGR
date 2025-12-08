import sys
import os
import torch
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))
from src.model.models_architectures import MODEL_ARCHITECTURES
from src.common.other_utils import extract_fusion_features, plot_fusion_feature_map, extract_timestamp

if __name__ == '__main__':
    model = MODEL_ARCHITECTURES['STMModel']()
    model_path = "models/2025-12-08_00-57_best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    sample = torch.randn(1, 1, 100)
    fusion = extract_fusion_features(model, sample)
    timestamp = extract_timestamp(model_path=model_path)
    plot_fusion_feature_map(fusion_features=fusion, timestamp=timestamp, save=True)
