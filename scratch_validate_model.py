import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=nn.Hardtanh):
        super().__init__()
        reduced_chs = int(in_chs * se_ratio)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = nn.Hardsigmoid(inplace=True) if hasattr(nn, 'Hardsigmoid') else nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)

class InvertedResidual(nn.Module):
    def __init__(self, in_chs, out_chs, dw_kernel=3, stride=1, exp_chs=None, act_layer=nn.Hardswish, has_se=False, se_reduced_chs=None):
        super().__init__()
        self.has_residual = (in_chs == out_chs and stride == 1)
        exp_chs = exp_chs or in_chs
        self.has_exp = exp_chs != in_chs

        if self.has_exp:
            self.conv_pw = nn.Conv2d(in_chs, exp_chs, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(exp_chs)
            self.act1 = act_layer(inplace=True)

        self.conv_dw = nn.Conv2d(exp_chs, exp_chs, dw_kernel, stride, dw_kernel // 2, groups=exp_chs, bias=False)
        self.bn2 = nn.BatchNorm2d(exp_chs)
        self.act2 = act_layer(inplace=True)

        if has_se:
            self.se = nn.Module()
            self.se.conv_reduce = nn.Conv2d(exp_chs, se_reduced_chs, 1, 1, 0, bias=True)
            self.se.act1 = nn.ReLU(inplace=True)
            self.se.conv_expand = nn.Conv2d(se_reduced_chs, exp_chs, 1, 1, 0, bias=True)
            self.se.gate_fn = nn.Hardsigmoid(inplace=True) if hasattr(nn, 'Hardsigmoid') else nn.Sigmoid()
            def se_forward(x):
                x_se = x.mean((2, 3), keepdim=True)
                x_se = self.se.conv_reduce(x_se)
                x_se = self.se.act1(x_se)
                x_se = self.se.conv_expand(x_se)
                return x * self.se.gate_fn(x_se)
            self.se.forward = se_forward
        else:
            self.se = nn.Identity()

        self.conv_pwl = nn.Conv2d(exp_chs, out_chs, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chs)

    def forward(self, x):
        res = x
        if self.has_exp:
            x = self.act1(self.bn1(self.conv_pw(x)))
        x = self.act2(self.bn2(self.conv_dw(x)))
        x = self.se(x)
        x = self.bn3(self.conv_pwl(x))
        if self.has_residual:
            x = x + res
        return x

class Block0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_dw = nn.Conv2d(16, 16, 3, 1, 1, groups=16, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_pw = nn.Conv2d(16, 16, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        res = x
        x = self.act1(self.bn1(self.conv_dw(x)))
        x = self.bn2(self.conv_pw(x))
        return x + res

class Block6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(160, 960, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(960)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        return self.act(self.bn1(self.conv(x)))

class MobileNetV3Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.Hardswish(inplace=True)

        HS = nn.Hardswish
        RE = nn.ReLU

        self.blocks = nn.Sequential(
            # Block 0
            nn.Sequential(Block0()),
            # Block 1
            nn.Sequential(
                InvertedResidual(16, 24, dw_kernel=3, stride=2, exp_chs=64, act_layer=RE, has_se=False),
                InvertedResidual(24, 24, dw_kernel=3, stride=1, exp_chs=72, act_layer=RE, has_se=False),
            ),
            # Block 2
            nn.Sequential(
                InvertedResidual(24, 40, dw_kernel=5, stride=2, exp_chs=72, act_layer=RE, has_se=True, se_reduced_chs=24),
                InvertedResidual(40, 40, dw_kernel=5, stride=1, exp_chs=120, act_layer=RE, has_se=True, se_reduced_chs=32),
                InvertedResidual(40, 40, dw_kernel=5, stride=1, exp_chs=120, act_layer=RE, has_se=True, se_reduced_chs=32),
            ),
            # Block 3
            nn.Sequential(
                InvertedResidual(40, 80, dw_kernel=3, stride=2, exp_chs=240, act_layer=HS, has_se=False),
                InvertedResidual(80, 80, dw_kernel=3, stride=1, exp_chs=200, act_layer=HS, has_se=False),
                InvertedResidual(80, 80, dw_kernel=3, stride=1, exp_chs=184, act_layer=HS, has_se=False),
                InvertedResidual(80, 80, dw_kernel=3, stride=1, exp_chs=184, act_layer=HS, has_se=False),
            ),
            # Block 4
            nn.Sequential(
                InvertedResidual(80, 112, dw_kernel=3, stride=1, exp_chs=480, act_layer=HS, has_se=True, se_reduced_chs=120),
                InvertedResidual(112, 112, dw_kernel=3, stride=1, exp_chs=672, act_layer=HS, has_se=True, se_reduced_chs=168),
            ),
            # Block 5
            nn.Sequential(
                InvertedResidual(112, 160, dw_kernel=5, stride=2, exp_chs=672, act_layer=HS, has_se=True, se_reduced_chs=168),
                InvertedResidual(160, 160, dw_kernel=5, stride=1, exp_chs=960, act_layer=HS, has_se=True, se_reduced_chs=240),
                InvertedResidual(160, 160, dw_kernel=5, stride=1, exp_chs=960, act_layer=HS, has_se=True, se_reduced_chs=240),
            ),
            # Block 6
            nn.Sequential(Block6()),
        )

        self.conv_head = nn.Conv2d(960, 1280, 1, 1, 0, bias=True)
        self.act2 = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x = self.blocks(x)
        x = self.act2(self.conv_head(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=30):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe

class VisualStream(nn.Module):
    def __init__(self, num_transformer_layers=1):
        super().__init__()
        self.cnn = MobileNetV3Large()
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.LayerNorm(256),
        )
        self.positional_encoding = PositionalEncoding(256, 30)
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_transformer_layers)

    def forward(self, x):
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)
        feats = self.feature_projection(feats)
        feats = feats.view(B, T, -1)
        feats = self.positional_encoding(feats)
        out = self.transformer(feats)
        return out.mean(dim=1)  # temporal pooling

class PoseMultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_branch, kernel_large=7, has_residual=False, out_channels=None):
        super().__init__()
        self.has_residual = has_residual
        self.branch_small = nn.Sequential(
            nn.Conv1d(in_channels, out_branch, kernel_size=1),
            nn.BatchNorm1d(out_branch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_branch, out_branch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_branch),
        )
        self.branch_large = nn.Sequential(
            nn.Conv1d(in_channels, out_branch, kernel_size=1),
            nn.BatchNorm1d(out_branch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_branch, out_branch, kernel_size=kernel_large, padding=kernel_large // 2, groups=out_branch),
            nn.BatchNorm1d(out_branch),
        )
        if has_residual and out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        s = self.branch_small(x)
        l = self.branch_large(x)
        out = torch.cat([s, l], dim=1)
        if self.has_residual:
            out = out + self.residual_proj(x)
        return F.relu(out)

class PoseStream(nn.Module):
    def __init__(self, num_transformer_layers=1):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(154, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.feature_extractor = nn.Sequential(
            PoseMultiScaleBlock(128, 64, kernel_large=7, has_residual=False),
            PoseMultiScaleBlock(128, 128, kernel_large=7, has_residual=True, out_channels=256),
        )
        self.to_transformer = nn.Linear(256, 256)
        self.positional_encoding = PositionalEncoding(256, 30)
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_transformer_layers)

    def forward(self, x):
        # x: (B, T, 154) -> transpose for 1D conv: (B, 154, T)
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.feature_extractor(x)
        # x: (B, 256, T) -> transpose back: (B, T, 256)
        x = x.transpose(1, 2)
        x = self.to_transformer(x)
        x = self.positional_encoding(x)
        out = self.transformer(x)
        return out.mean(dim=1)  # temporal pooling

class FusionModule(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.visual_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.pose_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.fusion_layer = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, v, p):
        gv = self.visual_gate(v)
        gp = self.pose_gate(p)
        fused = gv * v + gp * p
        return self.fusion_layer(fused)

class HierarchicalTwoStreamTransformer(nn.Module):
    def __init__(self, num_classes=8, num_transformer_layers=1):
        super().__init__()
        self.visual_stream = VisualStream(num_transformer_layers=num_transformer_layers)
        self.pose_stream = PoseStream(num_transformer_layers=num_transformer_layers)
        self.fusion = FusionModule(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, video_frames, pose_landmarks):
        v = self.visual_stream(video_frames)
        p = self.pose_stream(pose_landmarks)
        fused = self.fusion(v, p)
        return self.classifier(fused)

if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    import types, sys, pathlib
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

    model = HierarchicalTwoStreamTransformer(num_classes=8)
    msg = model.load_state_dict(sd, strict=True)
    print("Strict load status:", msg)
    print("Model verification succeeded! Running forward pass test...")

    sample_video = torch.zeros(1, 30, 3, 128, 128)
    sample_pose = torch.zeros(1, 30, 154)
    model.eval()
    with torch.no_grad():
        out = model(sample_video, sample_pose)
    print("Output shape:", out.shape)
    print("Softmax probs:", F.softmax(out, dim=-1).numpy())
