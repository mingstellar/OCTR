import ocnn
import torch
from mmcv.cnn import Scale, bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner.base_module import BaseModule
from torch import nn

from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models import HEADS, build_loss
from mmdet.core import reduce_mean

@HEADS.register_module()
class FCAF3DHeadOcnn(BaseModule):
    def __init__(
        self,
        n_classes,
        in_channels,
        out_channels,
        n_reg_outs,
        voxel_size,
        pts_prune_threshold,
        pts_assign_threshold,
        pts_center_threshold,
        center_loss=dict(type='CrossEntropyLoss', use_sigmoid=True),
        bbox_loss=dict(type='AxisAlignedIoULoss'),
        cls_loss=dict(type='FocalLoss'),
        train_cfg=None,
        test_cfg=None,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        self.voxel_size = voxel_size
        self.pts_prune_threshold = pts_prune_threshold
        self.pts_assign_threshold = pts_assign_threshold
        self.pts_center_threshold = pts_center_threshold
        self.center_loss = build_loss(center_loss)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)

    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        # init layers
        num = len(in_channels)
        self.out_blocks = torch.nn.ModuleList(
            [OutBlock(in_channels[i], out_channels) for i in range(num)]
        )
        self.up_blocks = torch.nn.ModuleList(
            [UpBlock(in_channels[i], in_channels[i - 1]) for i in range(1, num)]
        )

        # head layers
        self.center_head = torch.nn.Linear(out_channels, 1, bias=True)
        self.reg_head = torch.nn.Linear(out_channels, n_reg_outs, bias=True)
        self.cls_head = torch.nn.Linear(out_channels, n_classes, bias=True)
        # self.scales = nn.ModuleList([Scale(1.0) for _ in range(len(in_channels))])
    
    def init_weights(self):
        nn.init.normal_(self.center_head.weight.data, std=0.01)
        nn.init.normal_(self.reg_head.weight.data, std=0.01)
        nn.init.normal_(self.cls_head.weight.data, std=0.01)
        nn.init.constant_(self.cls_head.bias.data, bias_init_with_prob(0.01))

    def forward(self, inputs, octree, depths, pts_scale, center_list):
        center_preds, bbox_preds, cls_preds, points = [], [], [], []
        num = len(inputs)
        inputs = [ocnn.nn.octree_pad(inputs[i], octree, depths[i]) for i in range(num)]
        x = inputs[-1]
        for i in range(num - 1, -1, -1):
            if i < num - 1:
                x = self.up_blocks[i](x, octree, depths[i + 1])
                x = inputs[i] + x
                # x = self._prune(x, prune_score)

            out = self.out_blocks[i](x, octree, depths[i])
            center_pred, bbox_pred, cls_pred, point = \
            self._forward_single(out, octree, depths[i], pts_scale)
            center_preds.append(center_pred)
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)

        return center_preds[::-1], bbox_preds[::-1], cls_preds[::-1], points[::-1]

    def _forward_single(self, x, octree, depth, pts_scale):  # , reg_scale):
        centerness = self.center_head(x)
        cls_score = self.cls_head(x)
        regress = self.reg_head(x)
        # reg_distance = torch.exp(reg_scale(regress[:, :6]))
        reg_distance = torch.exp(regress[:, :6])
        reg_angle = regress[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)

        (x, y, z, b) = octree.xyzb(depth, nempty=False)
        xyz = torch.stack([x, y, z], dim=-1)
        xyz = ((xyz + 0.5) / (2 ** (depth - 1)) - 1) * pts_scale  # rescale points

        centernesses, bbox_preds, cls_scores, points = [], [], [], []
        for batch_id in range(octree.batch_size):
            batch_mask = b == batch_id
            centernesses.append(centerness[batch_mask])
            bbox_preds.append(bbox_pred[batch_mask])
            cls_scores.append(cls_score[batch_mask])
            points.append(xyz[batch_mask])

        return centernesses, bbox_preds, cls_scores, points
    
    def forward_train(self, x, octree, depths, pts_scale, center_list, gt_bboxes, gt_labels, input_metas):
        """Forward pass of the train stage.

        Args:
            x (torch.Tensor): Features from the backbone.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                bboxes of each sample.
            gt_labels(list[torch.Tensor]): Labels of each sample.
            input_metas (list[dict]): Contains scene meta info for each sample.

        Returns:
            dict: Centerness, bbox and classification loss values.
        """
        preds = self(x, octree, depths, pts_scale, center_list)
        return self._loss(*preds, gt_bboxes, gt_labels, input_metas)

    def forward_test(self, x, octree, depths, pts_scale, center_list, input_metas):
        """Forward pass of the test stage.

        Args:
            x (list[SparseTensor]): Features from the backbone.
            input_metas (list[dict]): Contains scene meta info for each sample.

        Returns:
            list[list[Tensor]]: bboxes, scores and labels for each sample.
        """
        preds = self(x, octree, depths, pts_scale, center_list)
        return self._get_bboxes(*preds, input_metas)
    
    def _loss(self, center_preds, bbox_preds, cls_preds, points, gt_bboxes,
              gt_labels, input_metas):
        """Per scene loss function.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes.
            gt_bboxes (list[BaseInstance3DBoxes]): Ground truth boxes for all
                scenes.
            gt_labels (list[Tensor]): Ground truth labels for all scenes.
            input_metas (list[dict]): Meta infos for all scenes.

        Returns:
            dict: Centerness, bbox, and classification loss values.
        """
        center_losses, bbox_losses, cls_losses = [], [], []
        for i in range(len(input_metas)):
            center_loss, bbox_loss, cls_loss = self._loss_single(
                center_preds=[x[i] for x in center_preds],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                input_meta=input_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i])
            center_losses.append(center_loss)
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
        return dict(
            center_loss=torch.mean(torch.stack(center_losses)),
            bbox_loss=torch.mean(torch.stack(bbox_losses)),
            cls_loss=torch.mean(torch.stack(cls_losses)))
    
    def _loss_single(self, center_preds, bbox_preds, cls_preds, points,
                     gt_bboxes, gt_labels, input_meta):
        """Per scene loss function.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.
            input_meta (dict): Scene meta info.

        Returns:
            tuple[Tensor]: Centerness, bbox, and classification loss values.
        """
        center_targets, bbox_targets, cls_targets = self._get_targets(
            points, gt_bboxes, gt_labels)

        center_preds = torch.cat(center_preds)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

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
    
    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.

        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.

        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
        y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
        z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5],
        ], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 0] + bbox_pred[:, 1] + \
            bbox_pred[:, 2] + bbox_pred[:, 3]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)

    def _get_bboxes(self, center_preds, bbox_preds, cls_preds, points,
                    input_metas):
        """Generate boxes for all scenes.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes.
            input_metas (list[dict]): Meta infos for all scenes.

        Returns:
            list[tuple[Tensor]]: Predicted bboxes, scores, and labels for
                all scenes.
        """
        results = []
        for i in range(len(input_metas)):
            result = self._get_bboxes_single(
                center_preds=[x[i] for x in center_preds],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                input_meta=input_metas[i])
            results.append(result)
        return results

    def _get_bboxes_single(self, center_preds, bbox_preds, cls_preds, points,
                           input_meta):
        """Generate boxes for a single scene.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            input_meta (dict): Scene meta info.

        Returns:
            tuple[Tensor]: Predicted bounding boxes, scores and labels.
        """
        mlvl_bboxes, mlvl_scores = [], []
        for center_pred, bbox_pred, cls_pred, point in zip(
                center_preds, bbox_preds, cls_preds, points):
            scores = cls_pred.sigmoid() * center_pred.sigmoid()
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]

            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes, scores, labels = self._single_scene_multiclass_nms(
            bboxes, scores, input_meta)
        return bboxes, scores, labels
    
    @staticmethod
    def _get_face_distances(points, boxes):
        """Calculate distances from point to box faces.

        Args:
            points (Tensor): Final locations of shape (N_points, N_boxes, 3).
            boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

        Returns:
            Tensor: Face distances of shape (N_points, N_boxes, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
        """
        shift = torch.stack(
            (points[..., 0] - boxes[..., 0], points[..., 1] - boxes[..., 1],
             points[..., 2] - boxes[..., 2]),
            dim=-1).permute(1, 0, 2)
        shift = rotation_3d_in_axis(
            shift, -boxes[0, :, 6], axis=2).permute(1, 0, 2)
        centers = boxes[..., :3] + shift
        dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
        dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
        dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                           dim=-1)

    @staticmethod
    def _get_centerness(face_distances):
        """Compute point centerness w.r.t containing box.

        Args:
            face_distances (Tensor): Face distances of shape (B, N, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).

        Returns:
            Tensor: Centerness of shape (B, N).
        """
        x_dims = face_distances[..., [0, 1]]
        y_dims = face_distances[..., [2, 3]]
        z_dims = face_distances[..., [4, 5]]
        centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
            y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
            z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
        return torch.sqrt(centerness_targets)
    
    @torch.no_grad()
    def _get_targets(self, points, gt_bboxes, gt_labels, levels=None):
        """Compute targets for final locations for a single scene.

        Args:
            points (list[Tensor]): Final locations for all levels.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.

        Returns:
            tuple[Tensor]: Centerness, bbox and classification
                targets for all locations.
        """
        float_max = points[0].new_tensor(1e8)
        n_levels = 4
        if levels is None:
            levels = torch.cat([
                points[i].new_tensor(i).expand(len(points[i]))
                for i in range(len(points))
            ])
            points = torch.cat(points)

        gt_bboxes = gt_bboxes.to(points.device)
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.unsqueeze(0).expand(n_points, n_boxes)

        # condition 1: point inside box
        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
                          dim=1)
        boxes = boxes.expand(n_points, n_boxes, 7)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        face_distances = self._get_face_distances(points, boxes)
        inside_box_condition = face_distances.min(dim=-1).values > 0

        # condition 2: positive points per level >= limit
        # calculate positive points per scale
        n_pos_points_per_level = []
        for i in range(n_levels):
            n_pos_points_per_level.append(
                torch.sum(inside_box_condition[levels == i], dim=0))
        # find best level
        n_pos_points_per_level = torch.stack(n_pos_points_per_level, dim=0)
        lower_limit_mask = n_pos_points_per_level < self.pts_assign_threshold
        lower_index = torch.argmax(lower_limit_mask.int(), dim=0) - 1
        lower_index = torch.where(lower_index < 0, 0, lower_index)
        all_upper_limit_mask = torch.all(
            torch.logical_not(lower_limit_mask), dim=0)
        best_level = torch.where(all_upper_limit_mask, n_levels - 1,
                                 lower_index)
        # keep only points with best level
        best_level = best_level.expand(n_points, n_boxes)
        levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = best_level == levels

        # condition 3: limit topk points per box by centerness
        centerness = self._get_centerness(face_distances)
        centerness = torch.where(inside_box_condition, centerness,
                                 torch.ones_like(centerness) * -1)
        centerness = torch.where(level_condition, centerness,
                                 torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(
            centerness,
            min(self.pts_center_threshold + 1, len(centerness)),
            dim=0).values[-1]
        topk_condition = centerness > top_centerness.unsqueeze(0)

        # condition 4: min volume box per point
        volumes = torch.where(inside_box_condition, volumes, float_max)
        volumes = torch.where(level_condition, volumes, float_max)
        volumes = torch.where(topk_condition, volumes, float_max)
        min_volumes, min_inds = volumes.min(dim=1)

        center_targets = centerness[torch.arange(n_points), min_inds]
        bbox_targets = boxes[torch.arange(n_points), min_inds]
        if not gt_bboxes.with_yaw:
            bbox_targets = bbox_targets[:, :-1]
        cls_targets = gt_labels[min_inds]
        cls_targets = torch.where(min_volumes == float_max, -1, cls_targets)
        return center_targets, bbox_targets, cls_targets

    def _single_scene_multiclass_nms(self, bboxes, scores, input_meta):
        """Multi-class nms for a single scene.

        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            input_meta (dict): Scene meta data.

        Returns:
            tuple[Tensor]: Predicted bboxes, scores and labels.
        """
        n_classes = scores.shape[1]
        with_yaw = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if with_yaw:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if with_yaw:
            box_dim = 7
        else:
            box_dim = 6
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = input_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels


class OutBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ocnn.nn.OctreeConv(
            in_channels, out_channels, kernel_size=[3], nempty=False
        )
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.elu = torch.nn.ELU()

    def forward(self, x, octree, depth):
        x = self.conv1(x, octree, depth)
        x = self.bn(x)
        x = self.elu(x)
        return x

class UpBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upconv1 = ocnn.nn.OctreeDeconv(
            in_channels, out_channels, kernel_size=[2], stride=2, nempty=False
        )
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.elu1 = torch.nn.ELU()
        self.conv2 = ocnn.nn.OctreeConv(
            out_channels, out_channels, kernel_size=[3], nempty=False
        )
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.elu2 = torch.nn.ELU()

    def forward(self, x, octree, depth):
        x = self.upconv1(x, octree, depth)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.conv2(x, octree, depth + 1)
        x = self.bn2(x)
        x = self.elu2(x)
        return x