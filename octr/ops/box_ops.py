import torch

def box_xyzwlh_to_xyzxyz(boxes):
    x, y, z, w, l, h = boxes.unbind(-1)
    b = [(x - 0.5 * w), (y - 0.5 * l), (z - 0.5 * h),
         (x + 0.5 * w), (y + 0.5 * l), (z + 0.5 * h)]
    return torch.stack(b, dim=-1)