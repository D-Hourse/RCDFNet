import torch


def get_reference_points_3d(H, W, Z=8, num_points_in_pillar=13, bs=1, device='cuda', dtype=torch.float):
    """Get the reference points used in HT.
    Args:
        H, W: spatial shape of bev.
        Z: hight of pillar.
        D: sample D points uniformly from each pillar.
        device (obj:`device`): The device where
            reference_points should be.
    Returns:
        Tensor: reference points used in HT, has \
            shape (bs, D, HW, 3).
    """
    zs_l = torch.linspace(3, Z - 1, 5, dtype=dtype, device=device)
    zs_g = torch.linspace(0.5, Z - 0.5, num_points_in_pillar - 5, dtype=dtype, device=device)
    zs = torch.cat((zs_l, zs_g)).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
    xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                        device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
    ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                        device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
    ref_3d = torch.stack((xs, ys, zs), -1)
    ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
    ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
    return ref_3d

voxel = get_reference_points_3d(160, 160)
print('done')