import sys
import types
import pathlib
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

pathlib.PosixPath = pathlib.WindowsPath

class AutoMod(types.ModuleType):
    def __getattr__(self, name):
        c = type(name, (), {})
        setattr(self, name, c)
        return c

sys.modules['configs'] = AutoMod('configs')
sys.modules['configs.config'] = AutoMod('configs.config')
sys.modules['configs.gating_config'] = AutoMod('configs.gating_config')

name = 'checkpoints/gating_model/best_model.pth'
p = hf_hub_download('Abs6187/isl-models', name)
ckpt = torch.load(p, map_location='cpu')
print(f"\n=== {name} ===")
print("Keys:", list(ckpt.keys()))
print("Epoch:", ckpt.get('epoch'))
print("Best val acc:", ckpt.get('best_val_acc'))
classes = ckpt.get('class_names')
print("Classes (count={}): {}".format(len(classes) if classes else 0, classes))
if 'model_state_dict' in ckpt:
    sd = ckpt['model_state_dict']
    print("Num params in gating model:", len(sd))
    print("Output layer shape:", [sd[k].shape for k in sd if 'classifier' in k or 'fc' in k or 'head' in k][-2:])
