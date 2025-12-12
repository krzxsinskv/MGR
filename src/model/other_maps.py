import sys
import os
import torch
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))
from src.model.models_architectures import MODEL_ARCHITECTURES
from src.common.other_utils import (extract_fusion_features, plot_fusion_feature_map, extract_timestamp,
                                    create_input_sample, load_case_config, plot_channel_attention_map,
                                    plot_post_attention_feature_map, plot_spatial_attention_map)

if __name__ == '__main__':
    config = load_case_config(case_number=1)
    model = MODEL_ARCHITECTURES[config["model"]["type"]]()
    model_path = config["evaluation"]["model"]
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    sample = create_input_sample()
    sample = torch.tensor(sample, dtype=torch.float32)
    sample = sample.unsqueeze(0).unsqueeze(0)

    _ = model(sample)

    timestamp = extract_timestamp(model_path=model_path)

    plot_channel_attention_map(model.captured["channel_attention_map"][0], timestamp)
    plot_spatial_attention_map(model.captured["spatial_attention_map"][0], timestamp)
    plot_post_attention_feature_map(model.captured["post_channel"][0], timestamp, stage="post_channel")
    plot_post_attention_feature_map(model.captured["post_spatial"][0], timestamp, stage="post_spatial")
