import torch
import torch_ext

dets = torch.Tensor([[49.1, 32.4, 51.0, 35.9],
                     [49.3, 32.9, 51.0, 35.3],
                     [35.3, 11.5, 39.9, 14.5],
                     [35.2, 11.7, 39.7, 15.7]])

scores = torch.Tensor([0.9, 0.9, 0.4, 0.3])


# static_nms_model
def static_nms_model(dets, scores):
    dets_ = torch.unsqueeze(dets, 0)
    scores_ = torch.unsqueeze(scores, 0)
    return torch_ext.static_nms(dets_, scores_, 0.6, -1.0, -1, -1, True)

outs = static_nms_model(dets, scores)
print("static_nms result:", outs)

trace = torch.jit.trace(static_nms_model, [dets, scores])
print(trace.graph)
trace.save("static_nms_model.pt")
torch.onnx.export(trace, [dets, scores], "static_nms_model.onnx", example_outputs=outs)

# static_batched_nms_model
idxs = torch.tensor([0, 0, 1, 1])
def static_batched_nms_model(dets, scores, idxs):
    dets_ = torch.unsqueeze(dets, 0)
    scores_ = torch.unsqueeze(scores, 0)
    idxs_ = torch.unsqueeze(idxs, 0)
    return torch_ext.static_batched_nms(dets_, scores_, idxs_, 0.6, -1.0, -1, -1, True)

outs = static_batched_nms_model(dets, scores, idxs)
print("static_nms result:", outs)

trace = torch.jit.trace(static_batched_nms_model, [dets, scores, idxs])
print(trace.graph)
trace.save("static_batched_nms_model.pt")
torch.onnx.export(trace, [dets, scores, idxs], "static_batched_nms_model.onnx", example_outputs=outs)


