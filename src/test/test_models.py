import sys
import os
import torch
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.path.join(sys.path[0], project_dir))
from src.model.models_architectures import MODEL_ARCHITECTURES
from src.common.other_utils import extract_fusion_features, plot_fusion_feature_map, extract_timestamp


def print_module_tree(mod):
    seen = {}
    def _walk(m, prefix=""):
        mid = id(m)
        seen[mid] = seen.get(mid, 0) + 1
        print(f"{prefix}{m.__class__.__name__} (id={mid})")
        for name, child in m.named_children():
            print(f"{prefix}  - {name}: ", end="")
            _walk(child, prefix + "    ")
    _walk(mod)
    dupes = {k:v for k,v in seen.items() if v>1}
    if dupes:
        print("\nWarning: these module object ids appear more than once (possible cycle or duplicate registration):")
        for mid,count in dupes.items():
            print(f" id={mid} appears {count} times")
    else:
        print("\nNo duplicate module object ids found.")


if __name__ == '__main__':
    model = MODEL_ARCHITECTURES['STMModel'](input_channels=1, seq_len=100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print("model moved to", device)
    # m = MODEL_ARCHITECTURES['STMModel'](input_channels=1, seq_len=10)
    # print_module_tree(m)

