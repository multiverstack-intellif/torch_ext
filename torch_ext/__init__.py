import os
import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

cur_path = os.path.dirname(os.path.realpath(__file__))
torch.ops.load_library(os.path.join(cur_path, "../lib/torch_ext.cpython-36m-x86_64-linux-gnu.so"))

# static nms
@parse_args("v", "v", "f", "f", "i", "i", "b")
def symbolic_static_nms(g, dets, scores, iou_threshold, score_threshold, max_output_size, top_k, invalid_to_bottom):
    iou_threshold = torch.tensor(iou_threshold, dtype=torch.float32)
    score_threshold = torch.tensor(score_threshold, dtype=torch.float32)
    max_output_size = torch.tensor(max_output_size, dtype=torch.int64)
    top_k = torch.tensor(top_k, dtype=torch.int64)
    invalid_to_bottom = torch.tensor(invalid_to_bottom, dtype=torch.bool)
    return g.op("opt_ops::static_nms", dets, scores, iou_threshold, score_threshold, max_output_size, top_k, invalid_to_bottom)

register_custom_op_symbolic("opt_ops::static_nms", symbolic_static_nms, 9)

def static_nms(dets, scores, iou_threshold, score_threshold, max_output_size, top_k, invalid_to_bottom):
    return torch.ops.opt_ops.static_nms(dets, scores, iou_threshold, score_threshold, max_output_size, top_k, invalid_to_bottom)

# static batched nms
@parse_args("v", "v", "v", "f", "f", "i", "i", "b")
def symbolic_static_batched_nms(g, dets, scores, idxs, iou_threshold, score_threshold, max_output_size, top_k, invalid_to_bottom):
    iou_threshold = torch.tensor(iou_threshold, dtype=torch.float32)
    score_threshold = torch.tensor(score_threshold, dtype=torch.float32)
    max_output_size = torch.tensor(max_output_size, dtype=torch.int64)
    top_k = torch.tensor(top_k, dtype=torch.int64)
    invalid_to_bottom = torch.tensor(invalid_to_bottom, dtype=torch.bool)
    return g.op("opt_ops::static_batched_nms", dets, scores, idxs, iou_threshold, score_threshold, max_output_size, top_k, invalid_to_bottom)

register_custom_op_symbolic("opt_ops::static_batched_nms", symbolic_static_batched_nms, 9)

def static_batched_nms(dets, scores, idxs, iou_threshold, score_threshold, max_output_size, top_k, invalid_to_bottom):
    return torch.ops.opt_ops.static_batched_nms(dets, scores, idxs, iou_threshold, score_threshold, max_output_size, top_k, invalid_to_bottom)
