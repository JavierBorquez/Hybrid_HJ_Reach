"""Optional whole-device NVIDIA memory sampling; no GPU dependency on CPU hosts."""
import csv
import shutil
import subprocess
import threading
import time


class GpuMemoryMonitor:
    def __init__(self, interval=.5):
        self.interval, self.samples = interval, []
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        if shutil.which('nvidia-smi'):
            self.start = time.perf_counter()
            def sample():
                while not self.stop.is_set():
                    try:
                        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                            '--format=csv,noheader,nounits'], text=True, timeout=3.)
                        used, total = map(float, output.splitlines()[0].split(','))
                        self.samples.append((time.perf_counter()-self.start, used, total))
                    except (OSError, subprocess.SubprocessError, ValueError):
                        pass
                    self.stop.wait(self.interval)
            self.thread = threading.Thread(target=sample, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=4.)

    def save(self, path):
        with open(path, 'w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['elapsed_s', 'total_device_used_mib', 'total_device_capacity_mib'])
            writer.writerows(self.samples)
        return max((s[1] for s in self.samples), default=None)
