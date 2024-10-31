import torch
import subprocess
import os
import sys
import psutil
import GPUtil
import time
from datetime import datetime


def check_gpu_setup():
    """Comprehensive GPU setup verification"""
    print("=== GPU Setup Diagnostic ===\n")

    # Check CUDA availability
    print("CUDA Availability:")
    print(f"CUDA is {'available' if torch.cuda.is_available() else 'NOT AVAILABLE'}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

    # GPU Information
    if torch.cuda.is_available():
        print("\nGPU Information:")
        current_device = torch.cuda.current_device()
        print(f"Current GPU: {torch.cuda.get_device_name(current_device)}")
        prop = torch.cuda.get_device_properties(current_device)
        print(f"Total memory: {prop.total_memory / 1024 ** 3:.1f} GB")
        print(f"CUDA capability: {prop.major}.{prop.minor}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Environment variables
    print("\nEnvironment Variables:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    # Detailed GPU info using nvidia-smi
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
        print("\nNVIDIA-SMI Output:")
        print(nvidia_smi)
    except:
        print("\nCouldn't run nvidia-smi. Make sure it's in your system PATH.")


def test_gpu_performance():
    """Run a simple GPU performance test"""
    if not torch.cuda.is_available():
        print("GPU is not available!")
        return

    print("\n=== GPU Performance Test ===")

    # Basic matrix multiplication test
    try:
        # Warm up GPU
        torch.cuda.empty_cache()

        # Create test tensors
        size = 5000
        print(f"\nRunning {size}x{size} matrix multiplication...")

        # Create tensors with appropriate memory size for 6GB VRAM
        size = 3000  # Reduced size for 6GB GPU
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')

        # Time the operation
        torch.cuda.synchronize()
        start_time = time.time()

        c = torch.matmul(a, b)

        torch.cuda.synchronize()
        end_time = time.time()

        print(f"Operation completed in {end_time - start_time:.2f} seconds")

        # Memory test
        print("\nMemory Test:")
        print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

        # Clean up
        del a, b, c
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during performance test: {str(e)}")


def monitor_gpu_usage(interval=1, duration=None):
    """
    Monitor GPU usage in real-time

    Args:
        interval (int): Update interval in seconds
        duration (int): How long to monitor in seconds (None for indefinite)
    """
    try:
        print("\n=== GPU Usage Monitor ===")
        print("Press Ctrl+C to stop monitoring\n")

        start_time = time.time()
        while True:
            # Clear screen (Windows)
            os.system('cls' if os.name == 'nt' else 'clear')

            # Get GPU information
            gpus = GPUtil.getGPUs()

            print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("\nGPU Stats:")
            print("-" * 50)

            for gpu in gpus:
                print(f"GPU ID: {gpu.id} ({gpu.name})")
                print(f"Load: {gpu.load * 100:.1f}%")
                print(f"Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.1f}%)")
                print(f"Temperature: {gpu.temperature}°C")
                print("-" * 50)

            # Also show PyTorch GPU memory stats
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024 ** 2
                reserved = torch.cuda.memory_reserved() / 1024 ** 2
                print("\nPyTorch GPU Memory:")
                print(f"Allocated: {allocated:.1f}MB")
                print(f"Reserved:  {reserved:.1f}MB")

            # Check if duration is reached
            if duration and (time.time() - start_time) > duration:
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError during monitoring: {str(e)}")


if __name__ == "__main__":
    # Run all diagnostics
    check_gpu_setup()
    test_gpu_performance()

    # Start monitoring (runs until Ctrl+C)
    monitor_gpu_usage(interval=1)