import torch
import subprocess
import os
import sys
import psutil
import GPUtil
import time
from datetime import datetime
import threading


class GPUMonitor:
    def __init__(self, log_interval=30):
        self.log_interval = log_interval
        self.running = False
        self.log_file = "gpu_monitor.log"

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def _monitor_loop(self):
        while self.running:
            try:
                # Get GPU stats
                gpu = GPUtil.getGPUs()[0]
                torch_allocated = torch.cuda.memory_allocated() / 1024 ** 2
                torch_reserved = torch.cuda.memory_reserved() / 1024 ** 2
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                ram_usage = process.memory_info().rss / 1024 ** 2

                # Create single-line status
                status = (
                    f"\rGPU: {gpu.load * 100:.0f}% | "
                    f"Mem: {gpu.memoryUsed}/{gpu.memoryTotal}MB | "
                    f"Temp: {gpu.temperature}°C | "
                    f"PyT: {torch_allocated:.0f}/{torch_reserved:.0f}MB | "
                    f"RAM: {ram_usage / 1024:.1f}GB"
                )

                # Clear line and print status
                print(status, end='', flush=True)

                # Log detailed information to file
                log_msg = (
                    f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    f"GPU Load: {gpu.load * 100:.1f}%\n"
                    f"GPU Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.1f}%)\n"
                    f"GPU Temperature: {gpu.temperature}°C\n"
                    f"PyTorch GPU Memory - Allocated: {torch_allocated:.1f}MB, Reserved: {torch_reserved:.1f}MB\n"
                    f"CPU Usage: {cpu_percent:.1f}%\n"
                    f"RAM Usage: {ram_usage:.1f}MB\n"
                )

                with open(self.log_file, 'a') as f:
                    f.write(log_msg)

            except Exception as e:
                print(f"\rMonitoring error: {str(e)}", end='', flush=True)

            time.sleep(self.log_interval)


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


if __name__ == "__main__":
    # Run diagnostic when script is run directly
    check_gpu_setup()