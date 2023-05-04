# IR-ADAS with FLIR ADK and Jetson Nano

## PyTorch model >> ONNX >> TensorRT
To speed-up inference time on Jetson Nano, we export our trained PyTorch model to [ONNX](https://onnx.ai) and finally to [TensorRT](https://developer.nvidia.com/tensorrt)

### Export PyTorch model to ONNX
#### Install requirements.txt
```sh
pip install -r requirements.txt
```

### Command to export PyTorch model to ONNX
```sh
python export.py --weights baseline.pt --grid
```

### Export ONNX to TensorRT engine (should be done on target device)
Jetson Nano with Jetpack 4.6 (Ubuntu 18.06, Python 3.6.9, CUDA 10.2)

#### Export PATH in `.bashrc`
```sh
export PATH=${PATH}:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.2/lib64
export CPATH=$CPATH:/usr/local/cuda-10.2/targets/aarch64-linux/include
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.2/targets/aarch64-linux/lib
```

```sh
$ source ~/.bashrc
```

#### Command to convert .onnx to .trt
```sh
$ \usr\src\tensorrt\bin\trtexec --onnx=baseline.onnx --saveEngine=baseline.trt --fp16 --verbose
```

## Inference on Jetson Nano
```
jetson_nano
├── engines
│   ├── baseline.trt
│   ├── README.md
├── utils
│   ├── __init__.py
│   ├── power.py
│   └── utils.py
├── README.md
├── requirements.txt
├── run-gui.py
└── run.py
```
### Installation
```sh
$ pip install -r requirements.txt
```

### Command to run inference on Jetson Nano
```sh
$ python run.py -e baseline.trt
```

## TODO
- [ ] : Backup Jetson Nano OS image with dependencies installed
- [x] : Implement shutdown from UPS on power failure
- [ ] : GUI for inference on Jetson Nano

## Reference
- https://github.com/WongKinYiu/yolov7