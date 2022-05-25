# Libtorch Model with object detection

![Example result video](https://github.com/kuobrian/libtorch_model/blob/json_version/images/demo.gif)

## How to use


### 1. Install libtorch

```
wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip

unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip

```

### 2. Change model weight path in model_setting.json


```
"Model": {
                "weights_path" : "../weights/cuda_yolov5s6.torchscript",
                "classes"      : "../weights/coco.names",
                ...
```
