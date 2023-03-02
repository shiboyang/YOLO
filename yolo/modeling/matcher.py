from typing import List

import torch

from detectron2.structures import Boxes


class Matcher:
    ignore_label_idx = 0
    negative_label_idx = 1
    positive_label_idx = 2

    def __init__(self, threshold: float, labels: List[int], allow_low_quality_matches: bool = False):
        self.threshold = threshold
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix, anchors: Boxes, gt_boxes: Boxes):
        """
        match_quality_matrix [K, H*W*A * num_feature_map]
        """
        assert match_quality_matrix.dim() == 2
        matched_val, matches = match_quality_matrix.max(dim=1)
        iou_mask = match_quality_matrix == matched_val[:, None]

        center_distance = self._calculate_l1_distance(gt_boxes, anchors)
        center_distance = center_distance.where(iou_mask, torch.tensor(float("inf"), device=anchors.device))
        distance_matched_val, center_matches = center_distance.min(dim=1)

        distance_mask = torch.full(center_distance.shape, False, device=anchors.device)
        distance_mask[range(len(gt_boxes)), center_matches] = True

        mask = (iou_mask & distance_mask)  # [K, HWA*num_feature_map]

        gt_idx, anchor_idx = torch.nonzero(mask, as_tuple=True)
        match_labels = matches.new_full((match_quality_matrix.shape[-1],), self.labels[self.negative_label_idx],
                                        dtype=torch.int8)
        match_labels[matched_val > self.threshold] = self.labels[self.ignore_label_idx]
        match_labels[anchor_idx] = self.labels[self.positive_label_idx]

        matches[anchor_idx] = gt_idx

        if self.allow_low_quality_matches:
            # todo
            raise NotImplementedError

        return matches, match_labels

    @staticmethod
    def _calculate_l1_distance(boxes1: Boxes, boxes2: Boxes):
        center_boxes1 = boxes1.get_centers()
        center_boxes2 = boxes2.get_centers()
        distance = torch.abs(center_boxes1[:, None, :] - center_boxes2[None, :, :]).sum(dim=-1)
        return distance
