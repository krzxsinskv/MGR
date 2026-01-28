import sys
import os
import torch
from pathlib import Path
import argparse

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))
from src.model.models_architectures import MODEL_ARCHITECTURES
from src.common.other_utils import (extract_fusion_features, plot_fusion_feature_map, extract_timestamp,
                                    create_input_sample, load_case_config, plot_channel_attention_map,
                                    plot_post_attention_feature_map, plot_spatial_attention_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdl", type=str, default=None,
                        help="Path to model .pth (overrides config if given)")
    args = parser.parse_args()

    model = MODEL_ARCHITECTURES["STMModel"]()
    model_path = args.mdl
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    sample = create_input_sample()
    sample = torch.tensor(sample, dtype=torch.float32)
    sample = sample.unsqueeze(0).unsqueeze(0)

    _ = model(sample)

    name = model_path.split("/")[-1].replace(".pth", "")

    plot_channel_attention_map(model.captured["channel_attention_map"][0], name)
    plot_spatial_attention_map(model.captured["spatial_attention_map"][0], name)
    plot_post_attention_feature_map(model.captured["post_channel"][0], name, stage="post_channel")
    plot_post_attention_feature_map(model.captured["post_spatial"][0], name, stage="post_spatial")
