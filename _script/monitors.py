import torch
import psutil
import GPUtil
import time
import threading
from datetime import datetime
import tqdm

class GPUMonitor:
    def __init__(self, log_interval=30, position=None):
        self.log_interval = log_interval
        self.position = position
        self.log_file = "gpu_monitor.log"
        self.running = False
        self.thread = None

    def start(self):
        """Start monitoring in a separate thread"""
        self.running = True
        print("\nGPU: Initializing...\n")  # Add extra newline
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                gpu = GPUtil.getGPUs()[0]
                torch_allocated = torch.cuda.memory_allocated() / 1024 ** 2
                torch_reserved = torch.cuda.memory_reserved() / 1024 ** 2
                process = psutil.Process()
                ram_usage = process.memory_info().rss / 1024 ** 2

                # Move cursor up two lines, print status, then move back
                print(f"\033[2A\rGPU: {gpu.load * 100:.0f}% | "
                      f"Mem: {gpu.memoryUsed}/{gpu.memoryTotal}MB | "
                      f"Temp: {gpu.temperature}°C | "
                      f"PyT: {torch_allocated:.0f}/{torch_reserved:.0f}MB | "
                      f"RAM: {ram_usage / 1024:.1f}GB\033[2B", end='', flush=True)

                # Log to file
                self._log_details(gpu, torch_allocated, torch_reserved, process)

            except Exception as e:
                print(f"\033[2A\rMonitor error: {str(e)}\033[2B", end='', flush=True)

            time.sleep(self.log_interval)

    def _log_details(self, gpu, torch_allocated, torch_reserved, process):
        """Log detailed statistics to file"""
        log_msg = (
            f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            f"GPU Load: {gpu.load * 100:.1f}%\n"
            f"GPU Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.1f}%)\n"
            f"GPU Temperature: {gpu.temperature}°C\n"
            f"PyTorch GPU Memory - Allocated: {torch_allocated:.1f}MB, Reserved: {torch_reserved:.1f}MB\n"
            f"CPU Usage: {process.cpu_percent():.1f}%\n"
            f"RAM Usage: {process.memory_info().rss / 1024 ** 2:.1f}MB\n"
        )

        with open(self.log_file, 'a') as f:
            f.write(log_msg)

    def set_position(self, lines_from_bottom):
        """Set the position where GPU stats should appear"""
        self.last_position = lines_from_bottom

    def log_status(self):
        # Use tqdm.write with position consideration
        status = self.get_status()
        if self.position is not None:
            # Move cursor to position before writing
            tqdm.write("\033[%dB%s\033[%dA" % (self.position, status, self.position))
        else:
            tqdm.write(status)