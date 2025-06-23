import os
import json
import time
import statistics
from datetime import datetime
from typing import Dict, List
from benchmark_models.system_monitor import SystemMonitor
from benchmark_models.token_counter import TokenCounter

class ModelBenchmark:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.token_counter = TokenCounter()
        self.system_monitor = SystemMonitor()
        
        # Model-specific tokenizer mappings
        self.model_tokenizer_map = {
            "llama2": "meta-llama/Llama-2-7b-hf",
            "llama3": "meta-llama/Meta-Llama-3-8B",
            "llama3.2": "meta-llama/Llama-3.2-3B",
            "mistral": "mistralai/Mistral-7B-v0.1",
            "codellama": "codellama/CodeLlama-7b-Python-hf",
            "phi": "microsoft/phi-2",
            "phi3": "microsoft/Phi-3-mini-4k-instruct",
            "gemma": "google/gemma-7b",
            "qwen": "Qwen/Qwen-7B",
            "qwen2.5": "Qwen/Qwen2.5-7B",
            "deepseek": "deepseek-ai/deepseek-coder-6.7b-base",
            "smollm2": "HuggingFaceTB/SmolLM2-1.7B",
        }
        
        self.models = {
            "smollm2:1.7b": "smollm2:1.7b",
            "deepseek-r1:1.5b": "deepseek-r1:1.5b", 
            "phi3:mini": "phi3:mini",  
            "gemma:2b": "gemma:2b",
            "llama3.2": "llama3.2",
            "qwen2.5:3b": "qwen2.5:3b",
        }
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens using the most appropriate method for the model"""
        model_base = model_name.split(':')[0].lower()
        
        # Try HuggingFace tokenizer first
        if model_base in self.model_tokenizer_map:
            hf_model = self.model_tokenizer_map[model_base]
            return self.token_counter.get_hf_tokenizer_count(text, hf_model)
        
        # Fallback to tiktoken for general use
        return self.token_counter.get_tiktoken_count(text)
    
    def _save_response_to_file(self, model_name: str, response: str) -> None:
        """Save the model response to a file in the benchmarks directory"""
        try:
            # Ensure benchmarks directory exists
            benchmarks_dir = "./benchmarks"
            os.makedirs(benchmarks_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Clean model name for filename (replace special characters)
            clean_model_name = model_name.replace(":", "-").replace("/", "-")
            
            # Create filename
            filename = f"response-{clean_model_name}-{timestamp}.txt"
            filepath = os.path.join(benchmarks_dir, filename)
            
            # Save response to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("="*50 + "\n\n")
                f.write(response)
                
            print(f"Response saved to: {filepath}")
            
        except Exception as e:
            print(f"Error saving response to file: {e}")
    
    def benchmark_with_streaming(self, model_name: str, prompt: str) -> Dict:
        """Benchmark with streaming and accurate token counting"""
        print(f"Benchmarking {model_name} with streaming...")
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        start_time = time.time()
        response_chunks = []
        chunk_times = []
        
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": True
                },
                stream=True,
                timeout=300  # 5 minute timeout
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data and data["response"]:
                            response_chunks.append(data["response"])
                            chunk_times.append(time.time())
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            end_time = time.time()
            full_response = "".join(response_chunks)
            
            # Stop system monitoring and get metrics
            system_metrics = self.system_monitor.stop_monitoring()
            
            # Save response to file
            self._save_response_to_file(model_name, full_response)
            
            # Count tokens accurately
            prompt_tokens = self.count_tokens(prompt, model_name)
            response_tokens = self.count_tokens(full_response, model_name)
            total_tokens = prompt_tokens + response_tokens
            
            total_duration = end_time - start_time
            
            # Calculate time to first token
            time_to_first_token = chunk_times[0] - start_time if chunk_times else 0
            
            # Calculate streaming metrics
            if len(chunk_times) > 1:
                streaming_duration = chunk_times[-1] - chunk_times[0]
                streaming_tokens_per_sec = response_tokens / streaming_duration if streaming_duration > 0 else 0
            else:
                streaming_tokens_per_sec = 0
            
            # Get model size information
            model_size_info = self.system_monitor.get_model_size(model_name)
            
            return {
                "model": model_name,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": total_tokens,
                "total_duration": total_duration,
                "time_to_first_token": time_to_first_token,
                "overall_tokens_per_second": total_tokens / total_duration if total_duration > 0 else 0,
                "response_tokens_per_second": response_tokens / total_duration if total_duration > 0 else 0,
                "streaming_tokens_per_second": streaming_tokens_per_sec,
                "chunks_received": len(response_chunks),
                "response_length_chars": len(full_response),
                
                # New system metrics
                "peak_memory_mb": system_metrics["peak_memory_mb"],
                "avg_cpu_percent": system_metrics["avg_cpu_percent"],
                "max_cpu_percent": system_metrics["max_cpu_percent"],
                "avg_gpu_utilization": system_metrics["avg_gpu_utilization"],
                "max_gpu_utilization": system_metrics["max_gpu_utilization"],
                
                # Model information
                "model_size_mb": model_size_info["size_mb"],
                "model_size_gb": model_size_info["size_gb"],
                "parameter_count": model_size_info["parameter_count"],
                "quantization": model_size_info["quantization"],
                
                # Quality metrics (placeholder)
                "response_quality_score": "TBC",
                "relevance_score": "TBC",
                
                "success": True
            }
            
        except Exception as e:
            # Stop monitoring in case of error
            self.system_monitor.stop_monitoring()
            return {"model": model_name, "error": str(e), "success": False}
    
    def benchmark_single_query(self, model_name: str, prompt: str, runs: int = 3) -> Dict:
        """Benchmark a single model with multiple runs using proper token counting"""
        results = []
        
        # Measure load time for the model (only once)
        print(f"  Measuring load time for {model_name}...")
        load_time = self.system_monitor.measure_load_time(model_name)
        
        for run in range(runs):
            print(f"  Run {run + 1}/{runs} for {model_name}")
            result = self.benchmark_with_streaming(model_name, prompt)
            if result.get("success", False):
                results.append(result)
            else:
                print(f"  ✗ Run {run + 1} failed: {result.get('error', 'Unknown error')}")
        
        if not results:
            return {"model": model_name, "error": "All runs failed", "success": False}
        
        # Calculate averages
        avg_result = {
            "model": model_name,
            "runs_completed": len(results),
            "runs_attempted": runs,
            "success_rate": len(results) / runs,
            "load_time_seconds": load_time,
            "avg_prompt_tokens": sum(r["prompt_tokens"] for r in results) / len(results),
            "avg_response_tokens": sum(r["response_tokens"] for r in results) / len(results),
            "avg_total_tokens": sum(r["total_tokens"] for r in results) / len(results),
            "avg_total_duration": sum(r["total_duration"] for r in results) / len(results),
            "avg_time_to_first_token": sum(r["time_to_first_token"] for r in results) / len(results),
            "avg_overall_tokens_per_second": sum(r["overall_tokens_per_second"] for r in results) / len(results),
            "avg_response_tokens_per_second": sum(r["response_tokens_per_second"] for r in results) / len(results),
            "avg_streaming_tokens_per_second": sum(r["streaming_tokens_per_second"] for r in results) / len(results),
            "min_tokens_per_second": min(r["response_tokens_per_second"] for r in results),
            "max_tokens_per_second": max(r["response_tokens_per_second"] for r in results),
            "std_tokens_per_second": statistics.stdev([r["response_tokens_per_second"] for r in results]) if len(results) > 1 else 0,
            
            # System metrics averages
            "avg_peak_memory_mb": sum(r["peak_memory_mb"] for r in results) / len(results),
            "max_peak_memory_mb": max(r["peak_memory_mb"] for r in results),
            "avg_cpu_percent": sum(r["avg_cpu_percent"] for r in results) / len(results),
            "max_cpu_percent": max(r["max_cpu_percent"] for r in results),
            "avg_gpu_utilization": sum(r["avg_gpu_utilization"] for r in results) / len(results),
            "max_gpu_utilization": max(r["max_gpu_utilization"] for r in results),
            
            # Model info (same for all runs)
            "model_size_mb": results[0]["model_size_mb"] if results else 0,
            "model_size_gb": results[0]["model_size_gb"] if results else 0,
            "parameter_count": results[0]["parameter_count"] if results else "Unknown",
            "quantization": results[0]["quantization"] if results else "Unknown",
            
            # Quality scores (placeholder)
            "response_quality_score": "TBC",
            "relevance_score": "TBC",
            
            "individual_results": results
        }
        
        return avg_result
    
    def run_benchmark(self, test_prompts: List[str], runs_per_prompt: int = 3):
        """Run benchmark across all models and prompts with token counting"""
        results = {}
        
        for model_name in self.models.keys():
            print(f"\n{'='*50}")
            print(f"Benchmarking {model_name}...")
            print('='*50)
            model_results = {}
            
            for i, prompt in enumerate(test_prompts):
                print(f"\nPrompt {i+1}/{len(test_prompts)}: {prompt[:50]}...")
                prompt_results = self.benchmark_single_query(model_name, prompt, runs_per_prompt)
                model_results[f"prompt_{i+1}"] = prompt_results
                
                if prompt_results.get("success_rate", 0) > 0:
                    print(f"  ✓ Avg tokens/sec: {prompt_results['avg_response_tokens_per_second']:.2f}")
                    print(f"  ✓ Time to first token: {prompt_results['avg_time_to_first_token']:.3f}s")
                    print(f"  ✓ Peak memory: {prompt_results['avg_peak_memory_mb']:.1f} MB")
                    print(f"  ✓ Load time: {prompt_results['load_time_seconds']:.2f}s")
                else:
                    print(f"  ✗ All runs failed")
                
            results[model_name] = model_results
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a formatted benchmark report with enhanced metrics"""
        report = "# Enhanced Model Benchmark Report\n\n"
        
        # Summary table with new metrics
        report += "## Summary\n\n"
        report += "| Model | Tokens/sec | TTFT (s) | Memory (MB) | CPU % | GPU % | Size (GB) | Load Time (s) | Quality | Relevance |\n"
        report += "|-------|------------|----------|-------------|-------|-------|-----------|---------------|---------|-----------|\n"
        
        for model_name, model_results in results.items():
            all_tokens_per_sec = []
            all_ttft = []
            all_memory = []
            all_cpu = []
            all_gpu = []
            all_load_time = []
            total_success = 0
            total_attempts = 0
            model_size_gb = 0
            
            for prompt_result in model_results.values():
                if prompt_result.get("success_rate", 0) > 0:
                    all_tokens_per_sec.append(prompt_result["avg_response_tokens_per_second"])
                    all_ttft.append(prompt_result["avg_time_to_first_token"])
                    all_memory.append(prompt_result["avg_peak_memory_mb"])
                    all_cpu.append(prompt_result["avg_cpu_percent"])
                    all_gpu.append(prompt_result["avg_gpu_utilization"])
                    all_load_time.append(prompt_result["load_time_seconds"])
                    total_success += prompt_result["runs_completed"]
                    total_attempts += prompt_result["runs_attempted"]
                    model_size_gb = prompt_result["model_size_gb"]
            
            avg_tokens_per_sec = statistics.mean(all_tokens_per_sec) if all_tokens_per_sec else 0
            avg_ttft = statistics.mean(all_ttft) if all_ttft else 0
            avg_memory = statistics.mean(all_memory) if all_memory else 0
            avg_cpu = statistics.mean(all_cpu) if all_cpu else 0
            avg_gpu = statistics.mean(all_gpu) if all_gpu else 0
            avg_load_time = statistics.mean(all_load_time) if all_load_time else 0
            
            report += f"| {model_name} | {avg_tokens_per_sec:.2f} | {avg_ttft:.3f} | {avg_memory:.1f} | {avg_cpu:.1f} | {avg_gpu:.1f} | {model_size_gb:.2f} | {avg_load_time:.2f} | TBC | TBC |\n"
        
        # Detailed results
        report += "\n## Detailed Results\n\n"
        for model_name, model_results in results.items():
            report += f"### {model_name}\n\n"
            for prompt_name, prompt_result in model_results.items():
                if prompt_result.get("success_rate", 0) > 0:
                    report += f"**{prompt_name}:**\n"
                    report += f"- Response Tokens/sec: {prompt_result['avg_response_tokens_per_second']:.2f}\n"
                    report += f"- Time to First Token: {prompt_result['avg_time_to_first_token']:.3f}s\n"
                    report += f"- Peak Memory Usage: {prompt_result['avg_peak_memory_mb']:.1f} MB\n"
                    report += f"- Average CPU Usage: {prompt_result['avg_cpu_percent']:.1f}%\n"
                    report += f"- Average GPU Usage: {prompt_result['avg_gpu_utilization']:.1f}%\n"
                    report += f"- Model Size: {prompt_result['model_size_gb']:.2f} GB\n"
                    report += f"- Load Time: {prompt_result['load_time_seconds']:.2f}s\n"
                    report += f"- Parameter Count: {prompt_result['parameter_count']}\n"
                    report += f"- Response Quality: {prompt_result['response_quality_score']}\n"
                    report += f"- Relevance Score: {prompt_result['relevance_score']}\n"
                    report += f"- Success Rate: {prompt_result['success_rate']:.1%}\n\n"
                else:
                    report += f"**{prompt_name}:** Failed - {prompt_result.get('error', 'Unknown error')}\n\n"
        
        return report
    
    def print_summary(self, results: Dict):
        """Print a comprehensive summary with enhanced metrics"""
        print("\n" + "="*100)
        print("ENHANCED MODEL BENCHMARK RESULTS")
        print("="*100)
        
        # Collect model performance data
        model_performance = []
        
        for model_name, model_results in results.items():
            all_tokens_per_sec = []
            all_ttft = []
            all_memory = []
            all_cpu = []
            all_gpu = []
            all_load_time = []
            total_success = 0
            total_attempts = 0
            model_size_gb = 0
            
            for prompt_result in model_results.values():
                if prompt_result.get("success_rate", 0) > 0:
                    all_tokens_per_sec.append(prompt_result["avg_response_tokens_per_second"])
                    all_ttft.append(prompt_result["avg_time_to_first_token"])
                    all_memory.append(prompt_result["avg_peak_memory_mb"])
                    all_cpu.append(prompt_result["avg_cpu_percent"])
                    all_gpu.append(prompt_result["avg_gpu_utilization"])
                    all_load_time.append(prompt_result["load_time_seconds"])
                    total_success += prompt_result["runs_completed"]
                    total_attempts += prompt_result["runs_attempted"]
                    model_size_gb = prompt_result["model_size_gb"]
            
            if all_tokens_per_sec:
                model_performance.append({
                    "model": model_name,
                    "avg_tokens_per_sec": statistics.mean(all_tokens_per_sec),
                    "avg_ttft": statistics.mean(all_ttft),
                    "avg_memory": statistics.mean(all_memory),
                    "avg_cpu": statistics.mean(all_cpu),
                    "avg_gpu": statistics.mean(all_gpu),
                    "avg_load_time": statistics.mean(all_load_time),
                    "model_size_gb": model_size_gb,
                    "success_rate": total_success / total_attempts
                })
        
        # Sort by tokens per second
        model_performance.sort(key=lambda x: x["avg_tokens_per_sec"], reverse=True)
        
        for rank, data in enumerate(model_performance, 1):
            print(f"\n{rank}. {data['model']}")
            print(f"   Tokens/sec: {data['avg_tokens_per_sec']:.2f}")
            print(f"   Time to first token: {data['avg_ttft']:.3f}s")
            print(f"   Peak memory: {data['avg_memory']:.1f} MB")
            print(f"   CPU usage: {data['avg_cpu']:.1f}%")
            print(f"   GPU usage: {data['avg_gpu']:.1f}%")
            print(f"   Model size: {data['model_size_gb']:.2f} GB")
            print(f"   Load time: {data['avg_load_time']:.2f}s")
            print(f"   Success rate: {data['success_rate']:.1%}")
        
        if not model_performance:
            print("\nNo successful benchmark results to display.")
        
        print("\n" + "="*100) 