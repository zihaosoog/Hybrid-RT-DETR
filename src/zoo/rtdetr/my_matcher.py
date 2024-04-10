"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

by lyuwenyu
"""

import torch
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

from src.core import register


@register
class DRMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.use_coeff_class = False


        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"


    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt):
        # cost = cost.T #[queries,tg]->[tg,queries]
        matching_matrix = torch.zeros_like(cost)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0
            del pos_idx

        del topk_ious, dynamic_ks#, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0.0
            matching_matrix[cost_argmin, multiple_match_mask] = 1.0
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        # matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        # gt_matched_classes = gt_classes[matched_gt_inds]

        # pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
        #     fg_mask_inboxes
        # ]
        # print('matching_matrix',matching_matrix.shape)
        
        return torch.nonzero(matching_matrix.T,as_tuple=True),num_fg
        # return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


    @torch.no_grad()
    def forward(self, outputs, targets, use_o2m):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits_o2o": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes_o2o": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries, num_class = outputs["pred_logits_o2o"].shape
        bs, num_queries_o2m, num_class = outputs["pred_logits_o2m"].shape
        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits_o2o"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits_o2o"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes_o2o"].flatten(0, 1)  # [batch_size * num_queries, 4]
        num_bbox_pre, _ = out_bbox.shape

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        num_gt,_ = tgt_bbox.shape

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class        
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        #TODO
        if self.use_coeff_class:
            gt_onehot_label = F.one_hot(tgt_ids,num_class).unsqueeze(0).repeat(num_bbox_pre,1,1)
            gt_soft_label = gt_onehot_label * (-cost_giou[...,None])
            _out_prob = outputs["pred_logits_o2o"].flatten(0, 1).softmax(-1)

            coeff_class = torch.dist(gt_soft_label, _out_prob.unsqueeze(1).repeat(1,num_gt,1), p=2)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        # print(sizes)
        indices_o2o = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        if use_o2m:
            indices_o2m = []
            for bs_idx, (out_prob_o2m, out_bbox_o2m, tgt_cls, tgt_reg) in enumerate(zip(outputs["pred_logits_o2m"], outputs["pred_boxes_o2m"], tgt_ids.split(sizes,-1), tgt_bbox.split(sizes,-2))):
                #TODO o2o more queries
                out_prob_o2m = F.sigmoid(out_prob_o2m)
                
                out_prob_o2m = out_prob_o2m[:, tgt_cls]
                neg_cost_class = (1 - self.alpha) * (out_prob_o2m**self.gamma) * (-(1 - out_prob_o2m + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob_o2m)**self.gamma) * (-(out_prob_o2m + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class

                # Compute the L1 cost between boxes
                cost_bbox = torch.cdist(out_bbox_o2m, tgt_reg, p=1)

                # Compute the giou cost betwen boxes
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox_o2m), box_cxcywh_to_xyxy(tgt_reg))

                C2m = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

                # print(C2m.T.shape, (-cost_giou).T.shape, tgt_cls.shape, sizes)
                C2m = C2m.view(num_queries_o2m, -1).cpu()
                
                indice_o2m_per = linear_sum_assignment(C2m)
                indices_o2m.append(tuple([torch.as_tensor(i, dtype=torch.int64) for i in indice_o2m_per]))

                #TODO o2m simota
                # pair_wise_ious,_ = box_iou(box_cxcywh_to_xyxy(tgt_reg), box_cxcywh_to_xyxy(out_bbox_o2m))
                # pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

                # gt_cls_per_image = (
                #     F.one_hot(tgt_cls.to(torch.int64), num_class)
                #     .float()
                #     .unsqueeze(1)
                #     .repeat(1, out_bbox_o2m.shape[0], 1)
                # )

                # cls_preds_ = (
                #     out_prob_o2m.float().unsqueeze(0).repeat(gt_cls_per_image.shape[0], 1, 1).sigmoid_()
                #     * out_prob_o2m.unsqueeze(0).repeat(gt_cls_per_image.shape[0], 1, 1).sigmoid_()
                # )
                # # 类别loss
                # pair_wise_cls_loss = F.binary_cross_entropy(
                #     cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
                # ).sum(-1)

                # C2m = (pair_wise_cls_loss
                #         + 3.0 * pair_wise_ious_loss
                #     )
                # # print(C2m.shape,pair_wise_ious.shape)
                # indice_o2m_per, num_fg = self.simota_matching(C2m, pair_wise_ious, tgt_cls, len(tgt_cls))
                # # # print('num_fg: ',num_fg)
                # # print('num_tgt: ',len(tgt_cls))
                # indices_o2m.append(indice_o2m_per)
                # del pair_wise_ious

            del indice_o2m_per, C2m, cost_bbox, cost_class, cost_giou
            # torch.cuda.empty_cache()
            return (indices_o2o, indices_o2m)
        else:
            return (indices_o2o, None)

if __name__=='__main__':
    import torch
    indice_o2m = (torch.tensor([1,2]),torch.tensor([3,4]))
    res = torch.stack(indice_o2m).permute(1,0)
    print(res)