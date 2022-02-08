// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
/* Modifications Copyright (c) Microsoft. */
#pragma once
#include <torch/script.h>


at::Tensor nms_cpu(const at::Tensor& dets,
                   const at::Tensor& scores,
                   const double threshold);

at::Tensor multi_label_nms_cpu(const at::Tensor& boxes,
                               const at::Tensor& scores,
                               const at::Tensor& max_output_boxes_per_class,
                               const at::Tensor& iou_threshold,
                               const at::Tensor& score_threshold);

at::Tensor static_nms_cpu(const at::Tensor& dets,
                      const at::Tensor& scores,
                      double iou_threshold,
                      double score_threshold,
                      int64_t max_output_size,
                      int64_t top_k,
                      bool invalid_to_bottom);

at::Tensor static_batched_nms_cpu(const at::Tensor& dets,
                              const at::Tensor& scores,
                              const at::Tensor& idxs,
                              double iou_threshold,
                              double score_threshold,
                              int64_t max_output_size,
                              int64_t top_k,
                              bool invalid_to_bottom);
