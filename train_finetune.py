import torch
import onnx
import os
import numpy as np
import onnxruntime as ort

# TensorRT imports
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 1. Load your PyTorch model
# Replace with your model class and checkpoint
from my_model import MyModel  # <-- your model here

model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 2. Dummy input and ONNX export
dummy_input = torch.randn(1, 3, 224, 224)  # Replace shape if needed
onnx_file = "model.onnx"
torch.onnx.export(model, dummy_input, onnx_file, input_names=['input'], output_names=['output'], opset_version=11)

print(f"Exported to ONNX: {onnx_file}")

# 3. Verify ONNX model
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")

# 4. Build TensorRT engine from ONNX
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_file, 'rb') as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise ValueError("Failed to parse ONNX")

builder.max_workspace_size = 1 << 30  # 1GB
builder.max_batch_size = 1
engine = builder.build_cuda_engine(network)
print("TensorRT engine built successfully.")

# 5. Save TensorRT engine
engine_path = "model.trt"
with open(engine_path, "wb") as f:
    f.write(engine.serialize())
print(f"TensorRT engine saved to {engine_path}")
