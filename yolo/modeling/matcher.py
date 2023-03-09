from typing import List, Tuple

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
        matched_val, _ = match_quality_matrix.max(dim=1)
        iou_mask = match_quality_matrix == matched_val[:, None]

        center_distance = self._calculate_l1_distance(gt_boxes, anchors)
        center_distance = center_distance.where(iou_mask, torch.tensor(float("inf"), device=anchors.device))
        distance_matched_val, center_matches = center_distance.min(dim=1)

        distance_mask = torch.full(center_distance.shape, False, device=anchors.device)
        distance_mask[range(len(gt_boxes)), center_matches] = True

        mask = (iou_mask & distance_mask)  # [K, HWA*num_feature_map]

        gt_idx, anchor_idx = torch.nonzero(mask, as_tuple=True)

        matched_val, matches = match_quality_matrix.max(dim=0)
        match_labels = torch.full(matches.size(), self.labels[self.negative_label_idx], dtype=torch.int8,
                                  device=matches.device)
        match_labels[matched_val > self.threshold] = self.labels[self.ignore_label_idx]
        match_labels[anchor_idx] = self.labels[self.positive_label_idx]

        matches[anchor_idx] = gt_idx

        if self.allow_low_quality_matches:
            raise NotImplementedError

        return matches, match_labels

    @staticmethod
    def _calculate_l1_distance(boxes1: Boxes, boxes2: Boxes):
        center_boxes1 = boxes1.get_centers()
        center_boxes2 = boxes2.get_centers()
        distance = torch.abs(center_boxes1[:, None, :] - center_boxes2[None, :, :]).sum(dim=-1)
        return distance


class Matcher2:
    ignore_label_idx = 0
    negative_label_idx = 1
    positive_label_idx = 2

    def __init__(self, threshold: float, labels: List[int], allow_low_quality_matches: bool = False):
        self.threshold = threshold
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: torch.Tensor, gt_boxes: Boxes, strides: List[int],
                 grid_sizes: List[Tuple[int, int]]):
        """
        match_quality_matrix: shape:[num_instance, num_anchor]
        boxes:


        """

        matched_val, matches = match_quality_matrix.max(dim=0)

        matched_labels = torch.full(matches.shape, self.labels[self.negative_label_idx], device=gt_boxes.device)
        matched_labels[matched_val > self.threshold] = self.labels[self.ignore_label_idx]

        pre_feature_length = 0
        for (feature_height, feature_width), stride in zip(grid_sizes, strides):
            mask = torch.full((3, feature_height, feature_width), False, device=gt_boxes.device)
            gt_box = gt_boxes.get_centers() / stride
            print(f"feature map w:{feature_width} h:{feature_height}")
            print("gt box in x maximum:", gt_box[:, [0]].max(), "minimum:", gt_box[:, [0]].min())
            print("gt box in y maximum:", gt_box[:, [1]].max(), "minimum:", gt_box[:, [1]].min())

            gt_box_idx = gt_box.long()
            mask[:, gt_box_idx[:, 1], gt_box_idx[:, 0]] = True
            mask = mask.view(-1)
            positive_mask_idx = torch.nonzero(mask).view(-1)
            positive_mask_idx += pre_feature_length
            matched_labels[positive_mask_idx] = self.labels[self.positive_label_idx]
            pre_feature_length += mask.numel()

        return matches, matched_labels
