#include <torch/extension.h>

template <typename T>
void get_valid_counts(
  const at::Tensor& dets,
  const at::Tensor& scores,
  double score_threshold,
  T* out_dets,
  int64_t* out_indices,
  int64_t* out_count
) {
  auto nbatch = dets.size(0);
  auto ndets = dets.size(1);
  auto box_data_len = dets.size(2);  // = 4

  for (int64_t _n = 0; _n < nbatch; _n++) {
    out_count[_n] = 0;
    for (int64_t _i = 0; _i < ndets; _i++) {
      auto score_val = scores[_n][_i].item().to<T>();
      if (score_val > score_threshold) {
          int64_t index_pass = _n * ndets + out_count[_n];
          for (int64_t _j = 0; _j < box_data_len; _j++) {
            out_dets[index_pass * box_data_len + _j] = dets[_n][_i][_j].item().to<T>();
          }
          out_indices[index_pass] = _i;
          out_count[_n]++;
      }

      if (_i >= out_count[_n]) {
          int64_t index_prev = _n * ndets + _i;
          for (int64_t _j = 0; _j < box_data_len; _j++) {
            out_dets[index_prev * box_data_len + _j] = -1.0;
          }
          out_indices[index_prev] = -1.0;
      }
    }
  }

}

//----------------------------------------------------------------------------------------
//                           nms_kernel_impl
// inputs:
//   dets (Tensor[nbatch, ndets, 4])) – boxes to perform NMS on
//                                      (x1, y1, x2, y2) format
//   scores (Tensor[nbatch, ndets]) – scores for each one of the boxes
//   ...
// outputs:
//   out_t(Tensor[nbatch, ndets, 6]) - the format 6 is (class_id, score, box_left, box_top,
//                                     box_right, box_bottom)
//   box_indices_t(Tensor[nbatch, ndets]) - box indices
//-----------------------------------------------------------------------------------------
template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor> nms_kernel_impl(
  const at::Tensor& dets,
  const at::Tensor& scores,
  const int64_t* idxs, // class id
  double iou_threshold,
  double score_threshold,
  int64_t max_output_size,
  bool force_suppress, // Whether to suppress all detections regardless of class_id
  int64_t top_k,
  bool invalid_to_bottom) {
  // AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  // AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  TORCH_CHECK(
      dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return std::make_tuple(at::empty({0}, dets.options()),
                           at::empty({0}, dets.options().dtype(at::kLong)));
  }

  auto nbatch = dets.size(0);
  auto ndets = dets.size(1);
  auto box_data_len = dets.size(2);
  at::Tensor valid_dets_t = at::zeros({nbatch, ndets, box_data_len}, dets.options());
  at::Tensor valid_indices_t = at::zeros({nbatch, ndets}, scores.options().dtype(at::kLong));
  at::Tensor valid_scores_t = at::zeros({nbatch, ndets}, scores.options());
  at::Tensor valid_classid_t = at::zeros({nbatch, ndets}, scores.options().dtype(at::kLong));
  at::Tensor valid_count_t = at::zeros({nbatch}, scores.options().dtype(at::kLong));
  auto valid_dets = valid_dets_t.data_ptr<scalar_t>();
  auto valid_indices = valid_indices_t.data_ptr<int64_t>();
  auto valid_scores = valid_scores_t.data_ptr<scalar_t>();
  auto valid_classid = valid_classid_t.data_ptr<int64_t>();
  auto valid_count = valid_count_t.data_ptr<int64_t>();

  if (score_threshold > 0.0) {
    get_valid_counts<scalar_t>(dets, scores, score_threshold, valid_dets, valid_indices, valid_count);
  } else {
    valid_dets_t = dets;
    for (int64_t _i = 0; _i < nbatch; _i++) {
      for (int64_t _j = 0; _j < ndets; _j++) {
        valid_indices[_i * ndets + _j] = _j;
      }
      valid_count[_i] = ndets;
    }
  }

  for (int64_t _i = 0; _i < nbatch; _i++) {
    for (int64_t _j = 0; _j < ndets; _j++) {
      auto score_index = valid_indices_t[_i][_j].item().to<int64_t>();
      int64_t index = _i * ndets + _j;
      if (_j < valid_count_t[_i].item().to<int64_t>()) {
        valid_scores[index] = scores[_i][score_index].item().to<scalar_t>();
        valid_classid[index] = idxs[_i * ndets + score_index];
      } else {
        valid_scores[index] = -1.0;
        valid_classid[index] = -1;
      }
    }
  }

  // for test:
  // std::cout << "valid_dets:" << std::endl << valid_dets_t << std::endl;
  // std::cout << "valid_scores: " << std::endl << valid_scores_t << std::endl;
  // std::cout << "valid_indices: " << std::endl << valid_indices_t << std::endl;
  // std::cout << "valid_classid: " << std::endl << valid_classid_t << std::endl;
  // std::cout << "valid_count: " << std::endl << valid_count_t << std::endl;

  at::Tensor order_t = at::zeros({nbatch, ndets}, scores.options().dtype(at::kLong));
  for (int64_t _i = 0 ; _i < nbatch; _i++) {
    // order_t[i] = std::get<1>(
    //     scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));  // used in new version!
    order_t[_i] = std::get<1>(
        valid_scores_t[_i].sort(/*dim=*/0, /* descending=*/true));
  }
  // auto order = order_t.data_ptr<int64_t>();
  // std::cout << "order: " << std::endl << order_t << std::endl;

  max_output_size = max_output_size == -1 ? ndets : max_output_size;
  if (max_output_size > ndets) {
    max_output_size = ndets;
  }

  int64_t det_info_len = box_data_len + 2;
  // default out_t: (class_id, score, box_left, box_top, box_right, box_bottom)
  at::Tensor out_t = at::zeros({nbatch, ndets, det_info_len}, dets.options());
  at::Tensor box_indices_t = at::zeros({nbatch, ndets}, scores.options().dtype(at::kLong));
  auto out = out_t.data_ptr<scalar_t>();
  auto box_indices = box_indices_t.data_ptr<int64_t>();
  auto batch_unit_len = ndets * det_info_len;

  int64_t nkeep;
  int64_t boxes_coord_start = 2; // consider coordn. start is 2!!
  int64_t class_id_pos = 0; // consider class id start is 0!!
  int64_t score_pos = 1; // consider score start is 1!!
  at::Tensor areas_t = at::zeros({nbatch, ndets}, dets.options());
  auto areas = areas_t.data_ptr<scalar_t>();
  for (int64_t _i = 0; _i < nbatch; _i++) {
    int64_t ibatch_valid_count = valid_count_t[_i].item().to<int64_t>();
    if (iou_threshold > 0) {
      if (ibatch_valid_count > 0) {
        nkeep = ibatch_valid_count;
        if ((top_k > 0) && (top_k < nkeep)) {
          nkeep = top_k;
        }

        for (int64_t _j = 0; _j < nkeep; _j++) {
          int64_t index_start = _i * batch_unit_len + _j * det_info_len;
          int64_t order_ = order_t[_i][_j].item().to<int64_t>();
          out[index_start + class_id_pos] = valid_classid[_i * ndets + order_];
          out[index_start + score_pos] = valid_scores_t[_i][order_].item().to<scalar_t>();
          for (int64_t _k = 0; _k < box_data_len; _k++) {
            out[index_start + _k + boxes_coord_start] = valid_dets_t[_i][order_][_k].item().to<scalar_t>();
          }
          box_indices[_i * ndets + _j] = order_;
        }

        if ((top_k > 0) && (top_k < ibatch_valid_count)) {
          for (int64_t _j = nkeep; _j < ibatch_valid_count; _j++) {
            int64_t index_start = _i * batch_unit_len + _j * det_info_len;
            for (int64_t _k = 0; _k < det_info_len; _k++) {
              out[index_start + _k] = -1.0;
            }
            box_indices[_i * ndets + _j] = -1;
          }
        }

        auto x1_t = out_t[_i].select(1, boxes_coord_start).contiguous();
        auto y1_t = out_t[_i].select(1, boxes_coord_start + 1).contiguous();
        auto x2_t = out_t[_i].select(1, boxes_coord_start + 2).contiguous();
        auto y2_t = out_t[_i].select(1, boxes_coord_start + 3).contiguous();
        auto x1 = x1_t.data_ptr<scalar_t>();
        auto y1 = y1_t.data_ptr<scalar_t>();
        auto x2 = x2_t.data_ptr<scalar_t>();
        auto y2 = y2_t.data_ptr<scalar_t>();

        for (int64_t _j = 0; _j < nkeep; _j++) {
          auto left = std::min(x1[_j], x2[_j]);
          auto top = std::min(y1[_j], y2[_j]);
          auto right = std::max(x1[_j], x2[_j]);
          auto bottom = std::max(y1[_j], y2[_j]);

          int64_t index_start = _i * batch_unit_len + _j * det_info_len + boxes_coord_start;
          out[index_start] = left;
          out[index_start + 1] = top;
          out[index_start + 2] = right;
          out[index_start + 3] = bottom;
          areas[_i * det_info_len + _j] = (right - left) * (bottom - top);
        }

      } // end of if (ibatch_valid_count > 0)

      // for test:
      // std::cout << "out_1: " << std::endl << out_t << std::endl;
      // std::cout << "box_indices_1: " << std::endl << box_indices_t << std::endl;
      // std::cout << "areas_1: " << std::endl << areas_t << std::endl;

      int64_t num_valid_boxes = 0; // use for afer nms
      bool is_valid_box;
      bool check_iou;

      for (int64_t _j = 0; _j < ibatch_valid_count; _j++) {
        int64_t index_j_start = _i * batch_unit_len + _j * det_info_len;
        if (num_valid_boxes == max_output_size) {  // if num_valid_boxes more than max_output_size:
          for (int64_t _k = 0; _k < det_info_len; _k++) {
            out[index_j_start + _k] = -1.0;
          }
          box_indices[_i * ndets + _j] = -1;
        } else if (out[index_j_start + score_pos] > 0) {
          is_valid_box = true;
          auto a_left = out[index_j_start + boxes_coord_start];
          auto a_top = out[index_j_start + boxes_coord_start + 1];
          auto a_right = out[index_j_start + boxes_coord_start + 2];
          auto a_bottom = out[index_j_start + boxes_coord_start + 3];
          auto jarea = areas[_j];

          for (int64_t _k = 0; _k < _j; _k++) {
            check_iou = false;
            int64_t index_k_start = _i * batch_unit_len + _k * det_info_len;
            if (is_valid_box && out[index_k_start + class_id_pos] >= 0) {
              if ((force_suppress) ||
                 (out[index_j_start + class_id_pos] == out[index_k_start + class_id_pos])) { // the same class id
                check_iou = true;
              }
            }

            if (check_iou) {
              auto b_left = out[index_k_start + boxes_coord_start];
              auto b_top = out[index_k_start + boxes_coord_start + 1];
              auto b_right = out[index_k_start + boxes_coord_start + 2];
              auto b_bottom = out[index_k_start + boxes_coord_start + 3];
              auto karea = areas[_k];

              auto w = std::max(static_cast<scalar_t>(0), std::min(a_right, b_right) - std::max(a_left, b_left));
              auto h = std::max(static_cast<scalar_t>(0), std::min(a_bottom, b_bottom) - std::max(a_top, b_top));
              auto inter_set = w * h;
              auto union_set = jarea + karea - inter_set;
              auto ovr = union_set <= 0 ? 0.0 : (inter_set * 1.0) / union_set;

              if (ovr >= iou_threshold) {
                is_valid_box = false;
              }
            }
          } // _K

          if (!is_valid_box) {
            for (int64_t _k = 0; _k < det_info_len; _k++) {
              out[index_j_start + _k] = -1.0;
            }
            box_indices[_i * ndets + _j] = -1;
          } else {
            num_valid_boxes++;
          }
        }
      } // end of for (int64_t _j = 0;...

    } else { // iou_threshold <= 0.0
      for (int64_t _j = 0; _j < ibatch_valid_count; _j++) {
        int64_t box_index = _i * ndets + _j;
        int64_t index = box_index * det_info_len;
        out[index] = valid_classid_t[_i][_j].item().to<int64_t>();
        out[index + 1] = valid_scores_t[_i][_j].item().to<scalar_t>();
        for (int64_t _k = 0; _k < box_data_len; _k++) {
            out[index + _k + 2] = valid_dets_t[_i][_j][_k].item().to<scalar_t>();
        }
        box_indices[box_index] = _j;
      }
    }

    // out of valid_count part
    for (int64_t _j = ibatch_valid_count; _j < ndets; _j++) {
      int64_t box_index = _i * ndets + _j;
      int64_t index = box_index * det_info_len;
      for (int64_t _k = 0; _k < det_info_len; _k++) {
          out[index + _k] = -1.0;
      }
      box_indices[box_index] = -1;
    }

    // raw box indices:
    for (int64_t _j = 0; _j < valid_count[_i]; _j++) {
      auto idx = box_indices_t[_i][_j].item().to<int64_t>();
      if (idx >= 0) {
        box_indices_t[_i][_j] = valid_indices_t[_i][idx];
      }
    }
  } // end of _i

  // rearrange boxes and indices, move all valid entries to top.
  if (invalid_to_bottom) {
    at::Tensor out_rearr_t = at::zeros({nbatch, ndets, det_info_len}, dets.options());
    at::Tensor box_indices_rearr_t = at::zeros({nbatch, ndets}, scores.options().dtype(at::kLong));
    auto out_rearr = out_rearr_t.data_ptr<scalar_t>();
    auto box_indices_rearr = box_indices_rearr_t.data_ptr<int64_t>();
    for (int64_t _i = 0; _i < nbatch; _i++) {
      int64_t valid_boxes_count = 0;
      for (int64_t _j = 0; _j < ndets; _j++) {
        int64_t box_index = _i * ndets + _j;
        int64_t index = box_index * det_info_len;
        int64_t box_index_rearr = _i * ndets + valid_boxes_count;
        int64_t index_rearr = box_index_rearr * det_info_len;
        if (out[index + class_id_pos] >= 0) {
          for (int64_t _k = 0; _k < det_info_len; _k++){
            out_rearr[index_rearr + _k] = out[index + _k];
          } // out_rearr_t[_i][valid_boxes_count] = out_t[_i][_j];
          box_indices_rearr[box_index_rearr] = box_indices[box_index];
          valid_boxes_count++;
        }

        if (_j >= valid_boxes_count) {
          for (int64_t _k = 0; _k < det_info_len; _k++){
            out_rearr[index + _k] = -1.0;
          }
          box_indices_rearr[box_index] = -1;
        }
      }
    }

    auto out_narrow_t = max_output_size < ndets ? out_rearr_t.narrow(1, 0, max_output_size) : out_rearr_t;
    return std::make_tuple(out_narrow_t, box_indices_rearr_t);
  }

  auto out_narrow_t = max_output_size < ndets ? out_t.narrow(1, 0, max_output_size) : out_t;
  return std::make_tuple(out_narrow_t, box_indices_t);

}

//--------------------------------------------------------------------
//                          static_nms
// inputs:
//   boxes (Tensor[batch_size, ndets, 4])) – boxes to perform NMS on
//                                           (x1, y1, x2, y2) format
//   scores (Tensor[batch_size, ndets]) – scores for each one of the boxes
//   iou_threshold - non-maximum suppression threshold
//   score_threshold - lower limit of score for valid bounding boxes
//   max_output_size - max number of output valid boxes for each instance(>0 or -1).
//                     Return all valid boxes if the value of max_output_size is -1
//   top_k - keep maximum top k detections before nms, -1 for no limit.
//   invalid_to_bottom - whether to move all valid bounding boxes to the top.
// outputs:
//   out_t (Tensor[batch_size, ndets, 5])) - (score, x1, y1, x2, y2) format
//--------------------------------------------------------------------
at::Tensor static_nms_cpu(
  const at::Tensor& dets,
  const at::Tensor& scores,
  double iou_threshold,
  double score_threshold=-1.0,
  int64_t max_output_size=-1,
  int64_t top_k=-1,
  bool invalid_to_bottom=false) {
  TORCH_CHECK(
      dets.dim() == 3, "boxes should be a 3d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(2) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(2));
  TORCH_CHECK(
      scores.dim() == 2,
      "scores should be a 2d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));
  TORCH_CHECK(
      dets.size(1) == scores.size(1),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(1),
      " and ",
      scores.size(1));

  auto result = std::make_tuple(at::empty({0}, dets.options()),
                                at::empty({0}, dets.options().dtype(at::kLong)));

  // remove batch_size dim expanding:
  // int64_t batch_size = 1; // without batchsize dim in pytorch, set batch_size = 1 for computing!
  // auto idxs_t = at::zeros({batch_size, dets.size(0)}, dets.options().dtype(at::kLong)); // the same class id
  // auto dets_ = dets.expand({batch_size, dets.size(0), dets.size(1)});
  // auto scores_ = scores.expand({batch_size, scores.size(0)});
  // auto idxs_ = idxs_t.data_ptr<int64_t>();

  int64_t batch_size = dets.size(0);
  auto idxs_t = at::zeros({batch_size, dets.size(0)}, dets.options().dtype(at::kLong)); // the same class id
  auto idxs_ = idxs_t.data_ptr<int64_t>();
  bool force_suppress = true; // true: inter-class NMs, should be true!

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "custom_nms", [&] {
    result = nms_kernel_impl<scalar_t>(dets, scores, idxs_, iou_threshold, score_threshold,
                                       max_output_size, force_suppress, top_k, invalid_to_bottom);
  });

  // note - compare for pytorch:
  // std::cout << "box_indices: " << std::endl << std::get<1>(result) << std::endl;

  return std::get<0>(result).narrow(2, 1, 5); //length = 5
}

//--------------------------------------------------------------------
//                           static_batched nms
// Each index value correspond to a category, and NMS will not be applied
// between elements of different categories.
// inputs:
//   boxes (Tensor[batch_size, ndets, 4])) – boxes to perform NMS on
//                                           (x1, y1, x2, y2) format
//   scores (Tensor[batch_size, ndets]) – scores for each one of the boxes
//   idxs (Tensor[batch_size, ndets]) -  indices of the categories for each one of
//                                       the boxes (class_id)
//   iou_threshold - non-maximum suppression threshold
//   score_threshold - lower limit of score for valid bounding boxes
//   max_output_size - max number of output valid boxes for each instance(>0 or -1).
//                     Return all valid boxes if the value of max_output_size is -1
//   top_k - keep maximum top k detections before nms, -1 for no limit.
//   invalid_to_bottom - whether to move all valid bounding boxes to the top.
// outputs:
//   out_t (Tensor[batch_size, ndets, 5])) - (class_id, score, x1, y1, x2, y2) format
//--------------------------------------------------------------------
at::Tensor static_batched_nms_cpu(
  const at::Tensor& dets,
  const at::Tensor& scores,
  const at::Tensor& idxs,
  double iou_threshold,
  double score_threshold=-1.0,
  int64_t max_output_size=-1,
  int64_t top_k=-1,
  bool invalid_to_bottom=false) {
  TORCH_CHECK(
      dets.dim() == 3, "boxes should be a 3d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(2) == 4,
      "boxes should have 4 elements in dimension 2, got ",
      dets.size(2));
  TORCH_CHECK(
      scores.dim() == 2,
      "scores should be a 2d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));
  TORCH_CHECK(
      dets.size(0) == idxs.size(0),
      "boxes and idxs should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      idxs.size(0));
  TORCH_CHECK(
      dets.size(1) == scores.size(1),
      "boxes and scores should have same number of elements in ",
      "dimension 1, got ",
      dets.size(1),
      " and ",
      scores.size(1));
  TORCH_CHECK(
      dets.size(1) == idxs.size(1),
      "boxes and idxs should have same number of elements in ",
      "dimension 1, got ",
      dets.size(1),
      " and ",
      idxs.size(1));

  auto result = std::make_tuple(at::empty({0}, dets.options()),
                                at::empty({0}, dets.options().dtype(at::kLong)));

  // remove batch_size dim expanding:
  // int64_t batch_size = 1; // without batchsize dim in pytorch, set batch_size = 1 for computing!
  // auto dets_ = dets.expand({batch_size, dets.size(0), dets.size(1)});
  // auto scores_ = scores.expand({batch_size, scores.size(0)});
  // // auto idxs_ = idxs_t.data_ptr<int64_t>();
  // auto idxs_ = idxs.expand({batch_size, idxs.size(0)}).data_ptr<int64_t>();  //.data<int64_t>();

  auto idxs_ = idxs.data_ptr<int64_t>();
  bool force_suppress = false; // true: inter-class NMs

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "static_batched_nms", [&] {
    result = nms_kernel_impl<scalar_t>(dets, scores, idxs_, iou_threshold, score_threshold,
                                       max_output_size, force_suppress, top_k, invalid_to_bottom);
  });

  // note - compare for pytorch:
  // std::cout << "box_indices: " << std::endl << std::get<1>(result) << std::endl;

  return std::get<0>(result);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("run", &static_nms_cpu, "nms run");
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &static_batched_nms_cpu, "batched_nms run");
}
