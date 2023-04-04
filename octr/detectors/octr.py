import ocnn
from ocnn.octree import Octree, Points

from mmdet3d.models import DETECTORS
from mmdet3d.models import build_backbone, build_head
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors import Base3DDetector


@DETECTORS.register_module()
class OcnnDeformableDETR(Base3DDetector):
    def __init__(
        self,
        backbone,
        head,
        voxel_size,
        depth,
        full_depth,
        pretrained=False,
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__()
        self.backbone = build_backbone(backbone)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)
        self.voxel_size = voxel_size
        self.depth = depth
        self.full_depth = full_depth
        self.scale_factor = self.voxel_size / (2 / (2**self.depth))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

    def init_weights(self, pretrained=None):
        self.backbone.init_weights()
        self.head.init_weights()

    def build_octree(self, raw_points):
        octrees, centers = [], []
        batch_size = len(raw_points)
        for batch_idx in range(batch_size):
            xyz = raw_points[batch_idx][:, :3]
            color = raw_points[batch_idx][:, 3:]
            center = 0
            # center = (xyz.min(axis=0)[0] + xyz.max(axis=0)[0]) / 2.0
            xyz = (xyz - center) / self.scale_factor
            point_cloud = Points(xyz, features=color)
            point_cloud.clip() # Clips the point cloud to [-1,0, 1.0]

            octree = Octree(self.depth, self.full_depth, device=xyz.device)
            octree.build_octree(point_cloud)
            octrees.append(octree)
            centers.append(center)

        octree = ocnn.octree.merge_octrees(octrees)
        octree.construct_all_neigh()
        octree_feature = ocnn.modules.InputFeature("F", nempty=True)  # P:xyz, F:rgb
        x = octree_feature(octree)
        return x, octree, centers

    def extract_feat(self, x, octree, img_metas, centers):
        r"""Extract features from points."""
        outs, depths = self.backbone(x, octree, octree.depth)
        x = self.neck_with_head(outs, octree, depths, self.scale_factor, centers)
        return x

    def forward_train(self, points, gt_bboxes_3d, gt_labels_3d, img_metas):
        x, octree, centers = self.build_octree(points)
        x = self.extract_feat(x, octree, img_metas, centers)
        losses = self.neck_with_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        r"""Test function without augmentation."""
        x, octree, centers = self.build_octree(points)
        x = self.extract_feat(x, octree, img_metas, centers)
        bbox_list = self.neck_with_head.get_bboxes(*x, img_metas, rescale=rescale)
        # Todo: shape[B, N, ...]
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        pass
