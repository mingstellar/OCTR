import ocnn
import torch
from ocnn.octree import Octree
from mmdet.core import reduce_mean
from mmdet3d.models import HEADS
from .fcaf3d_head_ocnn import FCAF3DHeadOcnn


@HEADS.register_module()
class FCAF3DHeadOcnnMS(FCAF3DHeadOcnn):
    def forward(self, inputs, octree, depths, pts_scale, center_list):
        num = len(inputs)
        max_depth = max(depths)
        inputs = [ocnn.nn.octree_pad(inputs[i], octree, depths[i]) for i in range(num)]

        outs = []
        x = inputs[-1]
        for i in range(num - 1, -1, -1):
            if i < num - 1:
                x = self.up_blocks[i](x, octree, depths[i + 1])
                x = inputs[i] + x
            out = self.out_blocks[i](x, octree, depths[i])
            key, xyz, batch = self.get_key_xyz(octree, depths[i], max_depth, pts_scale)
            if depths[i] != max_depth:
                mask = torch.logical_not(octree.nempty_mask(depths[i]))
                out = out[mask]
                key = key[mask]
                xyz = xyz[mask]
                batch = batch[mask]
            depth = torch.ones_like(batch) * depths[i]
            outs.append([out, key, xyz, batch, depth])

        out, key, xyz, batch, depth = [torch.cat(v, dim=0) for v in zip(*outs)]
        centerness = self.center_head(out)
        cls_score = self.cls_head(out)
        regress = self.reg_head(out)
        reg_distance = torch.exp(regress[:, :6])
        reg_angle = regress[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)

        centernesses, bbox_preds, cls_scores, points, depths = [], [], [], [], []
        for batch_id in range(octree.batch_size):
            batch_mask = batch == batch_id
            centernesses.append(centerness[batch_mask])
            bbox_preds.append(bbox_pred[batch_mask])
            cls_scores.append(cls_score[batch_mask])
            points.append(xyz[batch_mask])
            depths.append(depth[batch_mask])
        return centernesses, bbox_preds, cls_scores, points, depths

    def get_key_xyz(self, octree: Octree, depth: int, max_depth: int, pts_scale: float):
        key = octree.key(depth, nempty=False)
        (x, y, z, b) = ocnn.octree.key2xyz(key, depth)
        xyz = torch.stack([x, y, z], dim=-1)
        xyz = ((xyz + 0.5) / (2 ** (depth - 1)) - 1) * pts_scale  # rescale points

        key = key & ((1 << 48) - 1)  # clean the highest 16 bits
        key = (key << ((max_depth - depth) * 3)) | b  # rescale keys
        return key, xyz, b

    def _loss(self, center_preds, bbox_preds, cls_preds, points, depths, gt_bboxes, \
              gt_labels, input_metas):

        center_losses, bbox_losses, cls_losses = [], [], []
        for i in range(len(input_metas)):
            center_loss, bbox_loss, cls_loss = self._loss_single(
                center_preds=center_preds[i],
                bbox_preds=bbox_preds[i],
                cls_preds=cls_preds[i],
                points=points[i],
                depths=depths[i],
                input_meta=input_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i]
            )
            center_losses.append(center_loss)
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
        return dict(
            center_loss=torch.mean(torch.stack(center_losses)),
            bbox_loss=torch.mean(torch.stack(bbox_losses)),
            cls_loss=torch.mean(torch.stack(cls_losses))
        )

    # per scene
    def _loss_single(self, center_preds, bbox_preds, cls_preds, points, depths, \
                     gt_bboxes, gt_labels, input_meta):
    
        levels = depths.max() - depths
        center_targets, bbox_targets, cls_targets = self._get_targets(
            points, gt_bboxes, gt_labels, levels)

        # cls loss
        pos_inds = torch.nonzero(cls_targets >= 0).squeeze(1)
        n_pos = points.new_tensor(len(pos_inds))
        n_pos = max(reduce_mean(n_pos), 1.)
        cls_loss = self.cls_loss(cls_preds, cls_targets, avg_factor=n_pos)

        # bbox and centerness losses
        pos_center_preds = center_preds[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_center_targets = center_targets[pos_inds].unsqueeze(1)
        pos_bbox_targets = bbox_targets[pos_inds]
        # reduce_mean is outside if / else block to prevent deadlock
        center_denorm = max(
            reduce_mean(pos_center_targets.sum().detach()), 1e-6)
        if len(pos_inds) > 0:
            pos_points = points[pos_inds]
            center_loss = self.center_loss(
                pos_center_preds, pos_center_targets, avg_factor=n_pos)
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets),
                weight=pos_center_targets.squeeze(1),
                avg_factor=center_denorm)
        else:
            center_loss = pos_center_preds.sum()
            bbox_loss = pos_bbox_preds.sum()

        return center_loss, bbox_loss, cls_loss


    def _get_bboxes(self, center_preds, bbox_preds, cls_preds, points, depths, \
                    input_metas):
        results = []
        for i in range(len(input_metas)):
            result = self._get_bboxes_single(
                center_preds=center_preds[i],
                bbox_preds=bbox_preds[i],
                cls_preds=cls_preds[i],
                points=points[i],
                depths=depths[i],
                input_meta=input_metas[i])
            results.append(result)
        return results
    

    def _get_bboxes_single(self, center_preds, bbox_preds, cls_preds, points, depths, \
                           input_meta):

        scores = cls_preds.sigmoid() * center_preds.sigmoid()
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        bboxes = self._bbox_pred_to_bbox(points, bbox_preds)
        bboxes, scores, labels = self._single_scene_multiclass_nms(
            bboxes, scores, input_meta)
        return bboxes, scores, labels
