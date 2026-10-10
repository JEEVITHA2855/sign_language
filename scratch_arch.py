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

# List unique sub-block prefixes under visual_stream.cnn.blocks
block_names = sorted(set(['.'.join(k.split('.')[:4]) for k in sd if k.startswith('visual_stream.cnn.blocks')]))
print("CNN blocks found:")
for b in block_names:
    keys = [k for k in sd if k.startswith(b + '.')]
    shapes = {k[len(b)+1:]: tuple(sd[k].shape) for k in keys if 'weight' in k}
    print(f" {b}: weights={shapes}")
