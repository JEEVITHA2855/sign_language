import sys
import types
import pathlib
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

p = hf_hub_download('Abs6187/isl-models', 'best_model_new.pth')
ckpt = torch.load(p, map_location='cpu')
sd = ckpt['model_state_dict']

# Group parameters by module path
modules = {}
for k, v in sd.items():
    parts = k.split('.')
    mod_path = '.'.join(parts[:-1])
    param_name = parts[-1]
    if mod_path not in modules:
        modules[mod_path] = {}
    modules[mod_path][param_name] = tuple(v.shape)

for mod_path, params in sorted(modules.items()):
    # Skip cnn blocks to keep output concise
    if 'cnn.blocks' in mod_path:
        continue
    print(f"{mod_path}: {params}")
