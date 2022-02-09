# Extension for Pytorch, adding some optimized ops for deploy.

Try export torchscript and onnx with optimized ops:

```
  git clone https://github.com/pyjennings/torch_ext.git  # clone
  cd torch_ext
  python3 setup.py install  # build
  python3 test/test_nms.py  # Try export
```

Use in pytorch models. https://github.com/pyjennings/yolov5/tree/nms_deploy is an example of use static nms in yolov5. Install torch_ext as above and export torchscript/onnx:
```
  https://github.com/pyjennings/yolov5.git
  cd yolov5
  git checkout nms_deploy
  python3 export.py
```
