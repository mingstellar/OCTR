import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from mmdet3d.core import axis_aligned_bbox_overlaps_3d
from .box_ops import box_xyzwlh_to_xyzxyz

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_boxes, pred_logits, gt_boxes, gt_labels):
        """ Performs the matching

        Params:
            pred_boxes: Tensor of dim [batch, n_queries, 6] with the predicted box coordinates.
            pred_logits: Tensor of dim [batch, n_queries, num_classes] with the classification logits.
            gt_bboxes: List of Instances, (len(gt_bboxes) = batch_size)
            gt_labels: List of Tensors, (len(gt_labels) = batch_size)

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = pred_logits.shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = pred_logits.flatten(0, 1).sigmoid()
        out_bbox = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 6]

        # Also concat the target labels and boxes
        tgt_labels = torch.cat(gt_labels)
        tgt_boxes = torch.cat(gt_boxes)

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_labels] - neg_cost_class[:, tgt_labels]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_boxes[:, :6], p=1)

        # Compute the giou cost between boxes
        cost_giou = -axis_aligned_bbox_overlaps_3d(box_xyzwlh_to_xyzxyz(out_bbox),
                                                   box_xyzwlh_to_xyzxyz(tgt_boxes[:, :6]), mode='giou')

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in gt_boxes]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
