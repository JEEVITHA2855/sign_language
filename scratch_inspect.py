import sys
import types
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

class AutoMod(types.ModuleType):
    def __getattr__(self, name):
        c = type(name, (), {})
        setattr(self, name, c)
        return c

sys.modules['configs'] = AutoMod('configs')
sys.modules['configs.config'] = AutoMod('configs.config')
sys.modules['configs.gating_config'] = AutoMod('configs.gating_config')

for name in ['best_model_new.pth', 'checkpoints/gating_model/best_model.pth', 'checkpoints/group_0/best_model.pth', 'checkpoints/group_1/best_model.pth', 'checkpoints/group_2/best_model.pth']:
    try:
        p = hf_hub_download('Abs6187/isl-models', name)
        ckpt = torch.load(p, map_location='cpu')
        print(f"\n=== {name} ===")
        print("Keys:", list(ckpt.keys()))
        print("Epoch:", ckpt.get('epoch'))
        print("Best val acc:", ckpt.get('best_val_acc'))
        classes = ckpt.get('class_names')
        print("Classes (count={}): {}".format(len(classes) if classes else 0, classes))
    except Exception as e:
        print(f"Error loading {name}: {e}")
