此指南用于生成ocr engine并进行测试。

### 生成ocr engine

```
# 203
/home/nunova/tensorrt_tar/TensorRT-7.0.0.11/bin/trtexec --explicitBatch --onnx=crnn.onnx  --saveEngine=crnn.engine --fp16
# nx
/usr/src/tensorrt/bin/trtexec --explicitBatch --onnx=crnn.onnx  --saveEngine=crnn.engine --fp16
```

### 测试

```
mkdir build
cd build
cmake ..
make
./singal
```
/home/nunova/tensorrt_tar/TensorRT-7.0.0.11/bin/trtexec --explicitBatch --onnx=data/resnet181d.onnx  --saveEngine=data/resnet181d.engine --fp16


/home/nunova/tensorrt_tar/TensorRT-7.0.0.11/bin/trtexec --onnx=data/mobile0.25.onnx  --saveEngine=mobile0.25.engine --fp16