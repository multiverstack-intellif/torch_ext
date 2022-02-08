// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision.h"


at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const double threshold) {
  at::Tensor result = nms_cpu(dets, scores, threshold);
  return result;
}


at::Tensor multi_label_nms(const at::Tensor& boxes,
                           const at::Tensor& scores,
                           const at::Tensor& max_output_boxes_per_class,
                           const at::Tensor& iou_threshold,
                           const at::Tensor& score_threshold) {
  at::Tensor result = multi_label_nms_cpu(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);
  return result;
}

at::Tensor static_nms(const at::Tensor& dets,
                      const at::Tensor& scores,
                      double iou_threshold,
                      double score_threshold,
                      int64_t max_output_size,
                      int64_t top_k,
                      bool invalid_to_bottom) {
  at::Tensor result = static_nms_cpu(dets, scores, iou_threshold, score_threshold, max_output_size, top_k, invalid_to_bottom);
  return result;
}

at::Tensor static_batched_nms(const at::Tensor& dets,
                              const at::Tensor& scores,
                              const at::Tensor& idxs,
                              double iou_threshold,
                              double score_threshold,
                              int64_t max_output_size,
                              int64_t top_k,
                              bool invalid_to_bottom) {
  at::Tensor result = static_batched_nms_cpu(dets, scores, idxs, iou_threshold, score_threshold, max_output_size, top_k, invalid_to_bottom);
  return result;
}
