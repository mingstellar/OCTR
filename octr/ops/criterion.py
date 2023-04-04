import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import HungarianMatcher
from mmdet3d.models import AxisAlignedIoULoss
from .box_ops import box_xyzwlh_to_xyzxyz

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha:float=0.25, gamma:float=2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher()
        self.focal_alpha = focal_alpha
        self.iou_loss = AxisAlignedIoULoss()

    def loss_labels(self, pred_logits, gt_labels, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(gt_labels, indices)])
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([pred_logits.shape[0], pred_logits.shape[1], pred_logits.shape[2] + 1],
                                            dtype=pred_logits.dtype, layout=pred_logits.layout, device=pred_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(pred_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * pred_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, pred_logits, gt_labels):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = pred_logits
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v) for v in gt_labels], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, pred_boxes, gt_boxes, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(gt_boxes, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes[:, :6], reduction='none')

        loss_iou = self.iou_loss(box_xyzwlh_to_xyzxyz(src_boxes),
                                 box_xyzwlh_to_xyzxyz(target_boxes[:, :6]))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_iou'] = loss_iou.sum() / num_boxes

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def forward(self, pred_boxes, pred_logits, gt_bboxes, gt_labels, img_metas):
        """
        pred_boxes: Tensor of dim [batch, n_queries, 6] with the predicted box coordinates.
        pred_logits: Tensor of dim [batch, n_queries, num_classes] with the classification logits.
        gt_bboxes: List of Instances, (len(gt_bboxes) = batch_size)
        gt_labels: List of Tensors, (len(gt_labels) = batch_size)
        """
        gt_boxes = [v.tensor.to(next(iter(gt_labels)).device) for v in gt_bboxes]
        # gt_boxes = [v.tensor.to(pred_boxes.device) for v in gt_bboxes]
        indices = self.matcher(pred_boxes, pred_logits, gt_boxes, gt_labels)
        num_boxes = sum(len(v) for v in gt_labels)

        loss_labels = self.loss_labels(pred_logits, gt_labels, indices, num_boxes)
        loss_cardinality = self.loss_cardinality(pred_logits, gt_labels)

        loss_boxes = self.loss_boxes(pred_boxes, gt_boxes, indices, num_boxes)

        return {**loss_labels, **loss_cardinality, **loss_boxes}
    