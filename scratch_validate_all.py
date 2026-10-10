import sys
import types
import pathlib
import torch
from huggingface_hub import hf_hub_download
from scratch_validate_model import HierarchicalTwoStreamTransformer

pathlib.PosixPath = pathlib.WindowsPath
class AutoMod(types.ModuleType):
    def __getattr__(self, name):
        c = type(name, (), {})
        setattr(self, name, c)
        return c

sys.modules['configs'] = AutoMod('configs')
sys.modules['configs.config'] = AutoMod('configs.config')
sys.modules['configs.gating_config'] = AutoMod('configs.gating_config')

checkpoints = {
    "gating": ("checkpoints/gating_model/best_model.pth", 3, 2),
    "group_0": ("checkpoints/group_0/best_model.pth", 10, 1),
    "group_1": ("checkpoints/group_1/best_model.pth", 10, 1),
    "group_2": ("checkpoints/group_2/best_model.pth", 8, 1),
    "root": ("best_model_new.pth", 8, 1),
}

for name, (path, num_classes, num_layers) in checkpoints.items():
    p = hf_hub_download('Abs6187/isl-models', path)
    ckpt = torch.load(p, map_location='cpu')
    model = HierarchicalTwoStreamTransformer(num_classes=num_classes, num_transformer_layers=num_layers)
    res = model.load_state_dict(ckpt['model_state_dict'], strict=True)
    print(f"[{name}] (classes={num_classes}, layers={num_layers}): {res}")
