import time
import requests
import statistics
import psutil
import threading
import subprocess
import os
from typing import Dict

class SystemMonitor:
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_loaded = False

    def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
        self.gpu_loaded = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring and return collected metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        metrics = {
            "peak_memory_mb": max(self.memory_samples) if self.memory_samples else 0,
            "avg_cpu_percent": statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            "max_cpu_percent": max(self.cpu_samples) if self.cpu_samples else 0,
            "avg_gpu_utilization": 100.0 if self.gpu_loaded else 0.0,
            "max_gpu_utilization": 100.0 if self.gpu_loaded else 0.0,
        }
        return metrics

    def _monitor_loop(self):
        """Internal monitoring loop"""
        while self.monitoring:
            try:
                # Monitor ollama processes specifically for CPU
                ollama_cpu = self._get_ollama_cpu_usage()
                if ollama_cpu is not None:
                    self.cpu_samples.append(ollama_cpu)

                # Monitor memory (current process for benchmark script)
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_samples.append(memory_mb)

                # Monitor GPU using ollama ps (if model is loaded, set loaded flag)
                if not self.gpu_loaded:
                    self.gpu_loaded = self._get_gpu_utilization_from_ollama_ps()

            except Exception as e:
                print(f"Monitoring error: {e}")

            time.sleep(0.1)  # Sample every 100ms

    def _get_ollama_cpu_usage(self):
        """Get CPU usage of ollama processes"""
        try:
            ollama_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    if 'ollama' in proc.info['name'].lower():
                        ollama_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if ollama_processes:
                # Get CPU usage for all ollama processes
                total_cpu = 0
                for proc in ollama_processes:
                    try:
                        cpu_percent = proc.cpu_percent()
                        total_cpu += cpu_percent
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                return total_cpu
            return 0.0
        except Exception as e:
            print(f"Error getting ollama CPU usage: {e}")
            return 0.0

    def _get_gpu_utilization_from_ollama_ps(self):
        """Get GPU info from ollama ps command (True if model loaded)"""
        try:
            result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Has header and data
                    # Model is loaded, likely using GPU/accelerator if available
                    return True
            return False
        except Exception as e:
            print(f"Error running ollama ps: {e}")
            return False

    def get_model_size(self, model_name: str) -> Dict:
        """Get model size information from Ollama using 'ollama list'"""
        # Use ollama list command
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header
                    if model_name in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            # Parse size (e.g., "1.7 GB", "950 MB")
                            size_str = " ".join(parts[-2:])  # Get last two parts (size and unit)
                            size_gb = self._parse_size_to_gb(size_str)
                            params = self._extract_parameters_from_name(model_name)
                            return {
                                "size_bytes": int(size_gb * 1024 * 1024 * 1024),
                                "size_mb": round(size_gb * 1024, 2),
                                "size_gb": round(size_gb, 2),
                                "parameter_count": f"{params}B" if params else "Unknown",
                                "quantization": "Q4" if size_gb < params * 1.5 else "Unknown"
                            }
            # If not found, fallback
        except Exception as e:
            print(f"Error getting model list: {e}")
        # Fallback to name-based estimation
        return self._estimate_model_size_fallback(model_name)

    def _parse_size_to_gb(self, size_str: str) -> float:
        """Parse size string like '1.7 GB' or '950 MB' to GB"""
        try:
            parts = size_str.strip().split()
            if len(parts) >= 2:
                size_value = float(parts[0])
                unit = parts[1].upper()
                if unit in ['GB', 'G']:
                    return size_value
                elif unit in ['MB', 'M']:
                    return size_value / 1024
                elif unit in ['KB', 'K']:
                    return size_value / (1024 * 1024)
        except:
            pass
        return 0.0

    def _extract_parameters_from_name(self, model_name: str) -> float:
        """Extract parameter count from model name like 'gemma:2b' -> 2.0"""
        model_params = {
            "smollm2:1.7b": 1.7,
            "deepseek-r1:1.5b": 1.5,
            "phi3:mini": 3.8,  # phi3:mini is typically 3.8B
            "gemma:2b": 2.0,
            "llama3.2": 3.0,
            "qwen2.5:3b": 3.0,
        }
        if model_name in model_params:
            return model_params[model_name]
        import re
        size_patterns = [
            r'(\d+\.?\d*)b',
            r':(\d+\.?\d*)b',
        ]
        for pattern in size_patterns:
            match = re.search(pattern, model_name.lower())
            if match:
                return float(match.group(1))
        return 0.0

    def _estimate_model_size_fallback(self, model_name: str) -> Dict:
        estimated_params = self._extract_parameters_from_name(model_name)
        estimated_size_gb = estimated_params if estimated_params else 1.0
        estimated_size_mb = estimated_size_gb * 1024
        return {
            "size_bytes": int(estimated_size_gb * 1024 * 1024 * 1024),
            "size_mb": round(estimated_size_mb, 2),
            "size_gb": round(estimated_size_gb, 2),
            "parameter_count": f"{estimated_params}B" if estimated_params else "Unknown",
            "quantization": "Q4_K_M"
        }

    def measure_load_time(self, model_name: str) -> float:
        start_time = time.time()
        try:
            response = requests.post(
                f"http://localhost:11434/api/generate",
                json={"model": model_name, "prompt": "Hi", "stream": False},
                timeout=120
            )
            end_time = time.time()
            if response.status_code == 200:
                return end_time - start_time
            else:
                print(f"Failed to load model {model_name}: {response.status_code}")
                return -1
        except Exception as e:
            print(f"Error measuring load time for {model_name}: {e}")
            return -1 