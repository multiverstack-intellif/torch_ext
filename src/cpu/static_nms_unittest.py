import numpy as np
import torch
import torchvision
# import static_nms
import static_batched_nms

def _gen_rand_inputs(num_boxes):
    box_len = 4
    boxes = torch.rand(num_boxes, box_len, dtype=torch.float) * 0.5
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    scores = np.linspace(0, 1, num=num_boxes).astype("float32")
    np.random.shuffle(scores)
    idxs = np.random.randint(0, 2, (num_boxes,))
    return boxes, torch.from_numpy(scores), torch.from_numpy(idxs)

# # config according to pytorch
# print("test static_nms")
# print("="*80)
# print("test0")
# for num_boxes, iou_thres in [(10, 0.3), (20, 0.1)]:
#     print("="*80)
#     in_boxes, in_scores, _ = _gen_rand_inputs(num_boxes)
#     pytorch_res = torchvision.ops.nms(in_boxes, in_scores, iou_thres)
#     pytorch_res = torch.unsqueeze(pytorch_res, 0)
#     custom_res = static_nms.run(torch.unsqueeze(in_boxes, 0), torch.unsqueeze(in_scores, 0), iou_thres,
#                                 -1.0,  # score_threshold
#                                 -1,  # max_output_size
#                                 -1,  # top_k
#                                 True)  #  invalid_to_bottom
#     print("in_boxes:")
#     print(in_boxes)
#     print("-"*50)
#     print("in_scores:")
#     print(in_scores)
#     print("-"*50)
#     print("pytorch_res: ")
#     print(pytorch_res)
#     print("-"*50)
#     print("custom_res: ")
#     print(custom_res)

# # test score_threshold & invalid_to_bottom
# print("="*80)
# print("test1")
# for num_boxes, iou_thres in [(10, 0.3), (20, 0.1)]:
#     print("="*80)
#     in_boxes, in_scores, _ = _gen_rand_inputs(num_boxes)
#     pytorch_res = torchvision.ops.nms(in_boxes, in_scores, iou_thres)
#     pytorch_res = torch.unsqueeze(pytorch_res, 0)
#     custom_res = static_nms.run(torch.unsqueeze(in_boxes, 0), torch.unsqueeze(in_scores, 0), iou_thres,
#                                 0.3,  # score_threshold
#                                 -1,  # max_output_size
#                                 -1,  # top_k
#                                 False)  #  invalid_to_bottom
#     print("in_boxes:")
#     print(in_boxes)
#     print("-"*50)
#     print("in_scores:")
#     print(in_scores)
#     print("-"*50)
#     print("pytorch_res: ")
#     print(pytorch_res)
#     print("-"*50)
#     print("custom_res: ")
#     print(custom_res)

# # test top_k
# print("="*80)
# print("test2")
# for num_boxes, iou_thres in [(10, 0.3), (20, 0.1)]:
#     print("="*80)
#     in_boxes, in_scores, _ = _gen_rand_inputs(num_boxes)
#     pytorch_res = torchvision.ops.nms(in_boxes, in_scores, iou_thres)
#     pytorch_res = torch.unsqueeze(pytorch_res, 0)
#     custom_res = static_nms.run(torch.unsqueeze(in_boxes, 0), torch.unsqueeze(in_scores, 0), iou_thres,
#                                 -1.0,  # score_threshold
#                                 -1,  # max_output_size
#                                 3,  # top_k
#                                 True)  #  invalid_to_bottom
#     print("in_boxes:")
#     print(in_boxes)
#     print("-"*50)
#     print("in_scores:")
#     print(in_scores)
#     print("-"*50)
#     print("pytorch_res: ")
#     print(pytorch_res)
#     print("-"*50)
#     print("custom_res: ")
#     print(custom_res)

# # test max_output_size
# print("="*80)
# print("test3")
# for num_boxes, iou_thres in [(10, 0.3), (20, 0.1)]:
#     print("="*80)
#     in_boxes, in_scores, _ = _gen_rand_inputs(num_boxes)
#     pytorch_res = torchvision.ops.nms(in_boxes, in_scores, iou_thres)
#     pytorch_res = torch.unsqueeze(pytorch_res, 0)
#     custom_res = static_nms.run(torch.unsqueeze(in_boxes, 0), torch.unsqueeze(in_scores, 0), iou_thres,
#                                 -1.0,  # score_threshold
#                                 5,  # max_output_size
#                                 3,  # top_k
#                                 True)  #  invalid_to_bottom
#     print("in_boxes:")
#     print(in_boxes)
#     print("-"*50)
#     print("in_scores:")
#     print(in_scores)
#     print("-"*50)
#     print("pytorch_res: ")
#     print(pytorch_res)
#     print("-"*50)
#     print("custom_res: ")
#     print(custom_res)

# ============================================================================================
print("//"*80)
print("test custom_batch_nms")
print("test0")
for num_boxes, iou_thres in [(10, 0.3), (20, 0.1)]:
    print("="*80)
    in_boxes, in_scores, in_idxs = _gen_rand_inputs(num_boxes)
    pytorch_res = torchvision.ops.batched_nms(in_boxes, in_scores, in_idxs, iou_thres)
    pytorch_res = torch.unsqueeze(pytorch_res, 0)
    custom_res = static_batched_nms.run(torch.unsqueeze(in_boxes, 0), torch.unsqueeze(in_scores, 0),
                                        torch.unsqueeze(in_idxs, 0), iou_thres,
                                        -1.0,  # score_threshold
                                        -1,  # max_output_size
                                        -1,  # top_k
                                        True)  #  invalid_to_bottom
    print("in_boxes:")
    print(in_boxes)
    print("-"*50)
    print("in_scores:")
    print(in_scores)
    print("-"*50)
    print("in_idxs:")
    print(in_idxs)
    print("-"*50)
    print("pytorch_res: ")
    print(pytorch_res)
    print("-"*50)
    print("custom_res: ")
    print(custom_res)

# test score_threshold & invalid_to_bottom
print("="*80)
print("test1")
for num_boxes, iou_thres in [(10, 0.3), (20, 0.1)]:
    print("="*80)
    in_boxes, in_scores, in_idxs = _gen_rand_inputs(num_boxes)
    pytorch_res = torchvision.ops.batched_nms(in_boxes, in_scores, in_idxs, iou_thres)
    pytorch_res = torch.unsqueeze(pytorch_res, 0)
    custom_res = static_batched_nms.run(torch.unsqueeze(in_boxes, 0), torch.unsqueeze(in_scores, 0),
                                        torch.unsqueeze(in_idxs, 0), iou_thres,
                                        0.3,  # score_threshold
                                        -1,  # max_output_size
                                        -1,  # top_k
                                        False)  #  invalid_to_bottom
    print("in_boxes:")
    print(in_boxes)
    print("-"*50)
    print("in_scores:")
    print(in_scores)
    print("-"*50)
    print("in_idxs:")
    print(in_idxs)
    print("-"*50)
    print("pytorch_res: ")
    print(pytorch_res)
    print("-"*50)
    print("custom_res: ")
    print(custom_res)

# test top_k
print("="*80)
print("test2")
for num_boxes, iou_thres in [(10, 0.3), (20, 0.1)]:
    print("="*80)
    in_boxes, in_scores, in_idxs = _gen_rand_inputs(num_boxes)
    pytorch_res = torchvision.ops.batched_nms(in_boxes, in_scores, in_idxs, iou_thres)
    pytorch_res = torch.unsqueeze(pytorch_res, 0)
    custom_res = static_batched_nms.run(torch.unsqueeze(in_boxes, 0), torch.unsqueeze(in_scores, 0),
                                        torch.unsqueeze(in_idxs, 0), iou_thres,
                                        -1.0,  # score_threshold
                                        -1,  # max_output_size
                                        3,  # top_k
                                        True)  #  invalid_to_bottom
    print("in_boxes:")
    print(in_boxes)
    print("-"*50)
    print("in_scores:")
    print(in_scores)
    print("-"*50)
    print("in_idxs:")
    print(in_idxs)
    print("-"*50)
    print("pytorch_res: ")
    print(pytorch_res)
    print("-"*50)
    print("custom_res: ")
    print(custom_res)

# test max_output_size
print("="*80)
print("test3")
for num_boxes, iou_thres in [(10, 0.3), (20, 0.1)]:
    print("="*80)
    in_boxes, in_scores, in_idxs = _gen_rand_inputs(num_boxes)
    pytorch_res = torchvision.ops.batched_nms(in_boxes, in_scores, in_idxs, iou_thres)
    pytorch_res = torch.unsqueeze(pytorch_res, 0)
    custom_res = static_batched_nms.run(torch.unsqueeze(in_boxes, 0), torch.unsqueeze(in_scores, 0),
                                        torch.unsqueeze(in_idxs, 0), iou_thres,
                                        -1.0,  # score_threshold
                                        5,  # max_output_size
                                        3,  # top_k
                                        True)  #  invalid_to_bottom
    print("in_boxes:")
    print(in_boxes)
    print("-"*50)
    print("in_scores:")
    print(in_scores)
    print("-"*50)
    print("in_idxs:")
    print(in_idxs)
    print("-"*50)
    print("pytorch_res: ")
    print(pytorch_res)
    print("-"*50)
    print("custom_res: ")
    print(custom_res)