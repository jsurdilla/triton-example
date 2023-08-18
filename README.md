There are 3 models:
- add_sub - took that from https://github.com/triton-inference-server/python_backend/tree/main/examples/add_sub
- resnet50 - took that from https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/PyTorch 
- resnet50_wrapper - basically, I created a Python model similar to add_sub then called resnet50 to illustrate calling a non-Python model (using a non-Python backend) from a Python model. Instead of a `.pt` file, this can be `onnx`. In theory, we would do some pre and post processing around the inference call to resnet50.

## Running the models
Start the Triton docker container from the project root (parent of this README).
```sh
# If you have a compatible GPU
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.07-py3 

# Otherwise
#  -v $(pwd)/models:/models will mount your models folder into the container's /models folder
docker run -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/models:/models nvcr.io/nvidia/tritonserver:23.07-py3
```

Inside the container, start the models:
```sh
# Install dependencies first
pip install torch torchvision

tritonserver --model-repository=/models
```

You know it's successful if you see:
```
I0818 02:56:33.968928 195 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0818 02:56:33.969149 195 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0818 02:56:34.020472 195 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

## Testing
In a separate command line:
```
python test_add_sub.py

python test_resnet50.py

# Test the wrapped resnet50 model
python test_resnet50_wrapper.py
```