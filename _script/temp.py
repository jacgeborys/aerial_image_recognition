import onnxruntime as ort
print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())