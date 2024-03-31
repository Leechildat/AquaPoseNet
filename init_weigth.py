import torch
from models.myModel import LiteHRNet


def change_parameter_names(state_dict):
    def change_name(src_pattern, dst_pattern):
        for src_key in list(state_dict.keys()):
            if src_key.startswith(src_pattern):
                dst_key = src_key.replace(src_pattern, dst_pattern, 1)
                state_dict[dst_key] = state_dict.pop(src_key)

    # 执行需要的参数名称更改
    for i in range(100):
        for j in range(100):
            change_name(f"layers.{i}.blocks.{j}.norm.", f"0.layers.{i}.{j}.ln_1.")
            change_name(f"layers.{i}.blocks.{j}.op", f"0.layers.{i}.{j}.self_attention")


base_dim = 96
backbone=dict(
    in_channels=3,
    extra=dict(
        stem=dict(
            stem_channels=32,
            out_channels=32,
            expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_branches=(2, 3, 4),
            num_depths=(
                (2, 2),
                (2, 2, 9),
                (2, 2, 9, 2)
            ),
            module_type=('VSS', 'VSS', 'VSS'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_dims=(
                (base_dim, base_dim * 2),
                (base_dim, base_dim * 2, base_dim * 4),
                (base_dim, base_dim * 2, base_dim * 4, base_dim * 8),
            )),
        with_head=True,
    ))
model = LiteHRNet(**backbone).to('cuda')

weigth_path = 'models/pertrain/imagenet_1k/vssm_tiny_0230_ckpt_epoch_262.pth'
state_dict = model.state_dict()

checkpoint = torch.load(weigth_path)

for key in checkpoint['model'].keys():
    print(f'{key:40s}')  # 输出：stage0.0.norm.weight

change_parameter_names(checkpoint['model'])

for key, param in state_dict.items():
    for per_key, per_param in checkpoint['model'].items():
        if per_key in key:
            param.data = per_param.data

count = 0
for key, param in state_dict.items():
    for per_key, per_param in checkpoint['model'].items():
        if per_key in key:
            sssssx =  torch.allclose(param.data, per_param.data)
            if sssssx:
                count += 1

print(count)