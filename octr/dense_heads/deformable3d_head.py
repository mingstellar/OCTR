import ocnn
import torch
from torch import nn
from mmdet3d.models import HEADS, build_loss
from mmcv.cnn import bias_init_with_prob
from ocnn.octree import Octree
from.fcaf3d_head_ocnn import OutBlock, UpBlock
from .deformable_transformer import DeformableTransformer
from torch.nn.utils.rnn import pad_sequence
from ..ops import SetCriterion

@HEADS.register_module()
class Deformable3DHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_classes,
        n_reg_outs,
        voxel_size,
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # neck layers
        num = len(in_channels)
        self.out_blocks = torch.nn.ModuleList(
            [OutBlock(in_channels[i], out_channels) for i in range(num)]
        )
        self.up_blocks = torch.nn.ModuleList(
            [UpBlock(in_channels[i], in_channels[i - 1]) for i in range(1, num)]
        )

        self.transformer = DeformableTransformer(d_model=out_channels, n_heads=4,
            enc_n_points=2, dec_n_points=4)

        # head layers
        self.reg_head = torch.nn.Linear(out_channels, n_reg_outs, bias=True)
        self.cls_head = torch.nn.Linear(out_channels, n_classes, bias=True)

        self.criterion = SetCriterion(n_classes)

    def init_weights(self):
        nn.init.normal_(self.reg_head.weight.data, std=0.01)
        nn.init.normal_(self.cls_head.weight.data, std=0.01)
        nn.init.constant_(self.cls_head.bias.data, bias_init_with_prob(0.01))

    def get_key_xyz(self, octree: Octree, depth: int, max_depth: int, pts_scale: float):
        key = octree.key(depth, nempty=False)
        (x, y, z, b) = ocnn.octree.key2xyz(key, depth)
        xyz = torch.stack([x, y, z], dim=-1)
        xyz = ((xyz + 0.5) / (2 ** (depth - 1)) - 1) * pts_scale  # rescale points

        key = key & ((1 << 48) - 1)  # clean the highest 16 bits
        key = (key << ((max_depth - depth) * 3)) | b  # rescale keys
        return key, xyz, b

    def forward(self, inputs, octree, depths, pts_scale, center_list):
        num = len(inputs) # num_scale
        max_depth = max(depths)
        # 对4层的点进行填充
        inputs = [ocnn.nn.octree_pad(inputs[i], octree, depths[i]) for i in range(num)]

        outs = []
        x = inputs[-1]
        for i in range(num - 1, -1, -1):
            if i < num - 1:
                x = self.up_blocks[i](x, octree, depths[i + 1])
                x = inputs[i] + x
            out = self.out_blocks[i](x, octree, depths[i]) # out block才是最关键的，将非零传递到零元素。
            key, xyz, batch = self.get_key_xyz(octree, depths[i], max_depth, pts_scale)
            if depths[i] != max_depth:
                mask = torch.logical_not(octree.nempty_mask(depths[i]))
                out = out[mask]
                key = key[mask]
                xyz = xyz[mask]
                batch = batch[mask]
            depth = torch.ones_like(batch) * depths[i]
            outs.append([out, key, xyz, batch, depth])

        out, key, xyz, batch, depth = [torch.cat(v, dim=0) for v in zip(*outs)] # key is not used
        outs, keys, xyzs, depths = [], [], [], []
        for batch_id in range(octree.batch_size):
            batch_mask = batch == batch_id
            outs.append(out[batch_mask])
            keys.append(key[batch_mask])
            xyzs.append(xyz[batch_mask])
            depths.append(depth[batch_mask])
        
        outs = pad_sequence(outs, batch_first=True, padding_value=0)
        keys = pad_sequence(keys, batch_first=True, padding_value=0)
        xyzs = pad_sequence(xyzs, batch_first=True, padding_value=0)
        depths = pad_sequence(depths, batch_first=True, padding_value=0)
        masks = keys != 0
        # (B, n_queries, d_model)
        box_features = self.transformer(outs, xyzs, masks=masks)
        # (B, n_queries, n_classes)
        box_classes = self.cls_head(box_features)
        # (B, n_queries, n_reg_outs)
        # (x, y, z, w, l, h)
        box_coords = self.reg_head(box_features)

        return box_coords, box_classes
    
    def loss(self, pred_boxes, pred_logits, gt_bboxes, gt_labels, img_metas):
        return self.criterion(pred_boxes, pred_logits, gt_bboxes, gt_labels, img_metas)
    
    def get_bboxes(self, pred_boxes, pred_logits, img_metas, rescale):
        results = []
        for i in range(len(img_metas)):
            prob = pred_logits[i].sigmoid()
            topk_values, topk_indexes = torch.topk(prob.flatten(), 100)
            scores = topk_values
            topk_boxes = topk_indexes // prob.shape[1]
            labels = topk_indexes % prob.shape[1]
            boxes = torch.gather(pred_boxes[i], 0, topk_boxes.unsqueeze(-1).repeat(1, 6))
            # origin
            boxes = img_metas[i]["box_type_3d"](boxes, box_dim=6, with_yaw=False, origin=(0.5, 0.5, 0.5))
            results.append((boxes, scores, labels))

        return results