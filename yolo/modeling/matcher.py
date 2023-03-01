from typing import List

import torch

from detectron2.structures import Boxes


class Matcher:
    def __init__(self, thresholds: List[float], labels: List[int], allow_low_quality_matches: bool = False):
        thresholds = thresholds[:]
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))

        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix, anchors: Boxes, gt_boxes: Boxes):
        """
        match_quality_matrix [K, H*W*A * num_feature_map]
        """
        assert match_quality_matrix.dim() == 2
        matched_val, matches = match_quality_matrix.max(dim=0)
        iou_mask = match_quality_matrix == matched_val

        center_distance = self._calculate_l1_distance(gt_boxes, anchors)
        center_distance = center_distance.where(~iou_mask, torch.tensor(float("inf")))
        distance_matched_val, center_matches = center_distance.min(dim=0)

        distance_mask = torch.full(center_distance.shape, False)
        distance_mask[center_matches] = True

        mask = iou_mask == distance_mask  # [K, HWA*num_feature_map]

        gt_idx, anchor_idx = torch.nonzero(mask, as_tuple=True)
        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)


    @staticmethod
    def _calculate_l1_distance(boxes1: Boxes, boxes2: Boxes):
        center_boxes1 = boxes1.get_centers()
        center_boxes2 = boxes2.get_centers()
        distance = torch.abs(center_boxes2[-1, None, 2] - center_boxes1[None, -1, 2]).sum(dim=-1)
        return distance
