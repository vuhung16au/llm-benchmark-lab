# LLM Benchmark Metrics Collection System

This document describes how metrics are collected in the LLM benchmark system, covering performance, system resources, token counting, and model information.

## Overview

The benchmark system collects comprehensive metrics across multiple dimensions:

1. **Performance Metrics** - Speed and throughput measurements
2. **System Resource Metrics** - CPU, memory, and GPU utilization
3. **Token Metrics** - Accurate token counting for prompts and responses
4. **Model Information** - Size, parameters, and quantization details
5. **Quality Metrics** - Response quality and relevance (by human evaluation)

## Architecture

The system is organized into modular components:

```
benchmark_models/
├── main.py              # Entry point and orchestration
├── model_benchmark.py   # Core benchmarking logic
├── system_monitor.py    # System resource monitoring
├── token_counter.py     # Token counting utilities
└── test_prompts.py      # Test prompt definitions
```

## 1. Performance Metrics Collection

### Core Performance Measurements

Performance metrics are collected in the `ModelBenchmark.benchmark_with_streaming()` method:

#### Timing Measurements
- **Total Duration**: `end_time - start_time`
- **Time to First Token (TTFT)**: `chunk_times[0] - start_time`
- **Streaming Duration**: `chunk_times[-1] - chunk_times[0]`

#### Throughput Calculations
- **Overall Tokens/Second**: `total_tokens / total_duration`
- **Response Tokens/Second**: `response_tokens / total_duration`
- **Streaming Tokens/Second**: `response_tokens / streaming_duration`

#### Streaming Metrics
- **Chunks Received**: Number of response chunks from streaming
- **Response Length**: Character count of full response

### Load Time Measurement

Model load time is measured separately in `SystemMonitor.measure_load_time()`:

```python
def measure_load_time(self, model_name: str) -> float:
    start_time = time.time()
    response = requests.post(
        f"http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": "Hi", "stream": False},
        timeout=120
    )
    end_time = time.time()
    return end_time - start_time
```

## 2. System Resource Monitoring

### SystemMonitor Class

The `SystemMonitor` class provides real-time system resource monitoring:

#### Monitoring Thread
- Runs in a separate thread during benchmarking
- Samples every 100ms (`time.sleep(0.1)`)
- Collects CPU, memory, and GPU metrics

#### CPU Monitoring
```python
def _get_ollama_cpu_usage(self):
    # Finds all ollama processes
    # Sums CPU usage across all ollama processes
    # Returns total CPU percentage
```

#### Memory Monitoring
```python
# Monitors current process memory (benchmark script)
process = psutil.Process()
memory_info = process.memory_info()
memory_mb = memory_info.rss / 1024 / 1024
```

#### GPU Monitoring
```python
def _get_gpu_utilization_from_ollama_ps(self):
    # Uses 'ollama ps' command to detect if model is loaded
    # Sets gpu_loaded flag if model is active
    # Returns True/False for GPU utilization
```

#### Collected System Metrics
- **Peak Memory (MB)**: Maximum memory usage during benchmark
- **Average CPU %**: Mean CPU usage across all samples
- **Max CPU %**: Peak CPU usage during benchmark
- **Average GPU %**: 100% if model loaded, 0% otherwise
- **Max GPU %**: Same as average (binary detection)

## 3. Token Counting System

### TokenCounter Class

The system uses multiple token counting methods for accuracy:

#### HuggingFace Tokenizers (Primary)
```python
def get_hf_tokenizer_count(self, text: str, model_name: str) -> int:
    # Loads model-specific tokenizer from HuggingFace
    # Uses trust_remote_code=True for custom tokenizers
    # Returns accurate token count for specific model
```

#### Model-Specific Tokenizer Mappings
```python
model_tokenizer_map = {
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
```

#### Tiktoken Fallback (Secondary)
```python
def get_tiktoken_count(self, text: str, model: str = "gpt-3.5-turbo") -> int:
    # Uses OpenAI's tiktoken library
    # Falls back to cl100k_base encoding if model not found
    # Provides general-purpose token counting
```

#### Fallback Counting (Tertiary)
```python
def _fallback_count(self, text: str) -> int:
    # Counts words + punctuation as tokens
    # More accurate than char/4 approximation
    # Used when tokenizers fail to load
```

### Token Counting Strategy

The system uses a hierarchical approach:

1. **Model-Specific**: Try HuggingFace tokenizer for exact model
2. **General Purpose**: Fall back to tiktoken for OpenAI-compatible counting
3. **Approximation**: Use word + punctuation counting as last resort

## 4. Model Information Collection

### Model Size Detection

The `SystemMonitor.get_model_size()` method extracts model information:

#### Ollama List Command
```python
def get_model_size(self, model_name: str) -> Dict:
    # Runs 'ollama list' command
    # Parses size information (e.g., "1.7 GB", "950 MB")
    # Extracts parameter count from model name
    # Estimates quantization based on size vs parameters
```

#### Size Parsing
```python
def _parse_size_to_gb(self, size_str: str) -> float:
    # Converts "1.7 GB", "950 MB", etc. to GB
    # Handles different unit formats
```

#### Parameter Extraction
```python
def _extract_parameters_from_name(self, model_name: str) -> float:
    # Extracts parameter count from model names
    # Handles patterns like "gemma:2b" -> 2.0
    # Uses predefined mappings for known models
```

#### Collected Model Information
- **Size (bytes, MB, GB)**: Model file size on disk
- **Parameter Count**: Estimated number of parameters
- **Quantization**: Estimated quantization level (Q4, etc.)

## 5. Data Flow and Aggregation

### Single Run Metrics

Each benchmark run collects:

```python
{
    "model": model_name,
    "prompt_tokens": prompt_tokens,
    "response_tokens": response_tokens,
    "total_tokens": total_tokens,
    "total_duration": total_duration,
    "time_to_first_token": time_to_first_token,
    "overall_tokens_per_second": total_tokens / total_duration,
    "response_tokens_per_second": response_tokens / total_duration,
    "streaming_tokens_per_second": streaming_tokens_per_sec,
    "chunks_received": len(response_chunks),
    "response_length_chars": len(full_response),
    "peak_memory_mb": system_metrics["peak_memory_mb"],
    "avg_cpu_percent": system_metrics["avg_cpu_percent"],
    "max_cpu_percent": system_metrics["max_cpu_percent"],
    "avg_gpu_utilization": system_metrics["avg_gpu_utilization"],
    "max_gpu_utilization": system_metrics["max_gpu_utilization"],
    "model_size_mb": model_size_info["size_mb"],
    "model_size_gb": model_size_info["size_gb"],
    "parameter_count": model_size_info["parameter_count"],
    "quantization": model_size_info["quantization"],
    "response_quality_score": "TBC",
    "relevance_score": "TBC",
    "success": True
}
```

### Multi-Run Aggregation

The `benchmark_single_query()` method aggregates multiple runs:

#### Statistical Aggregation
- **Averages**: Mean values across all successful runs
- **Min/Max**: Extreme values for performance metrics
- **Standard Deviation**: Variability in tokens/second
- **Success Rate**: Percentage of successful runs

#### System Metrics Aggregation
- **Average Peak Memory**: Mean peak memory across runs
- **Max Peak Memory**: Highest memory usage observed
- **Average/Max CPU**: CPU usage statistics
- **Average/Max GPU**: GPU utilization statistics

### Cross-Model Aggregation

The `run_benchmark()` method aggregates across models and prompts:

#### Summary Statistics
- **Overall Performance**: Average across all prompts
- **Consistency**: Standard deviation across prompts
- **Reliability**: Success rates across different scenarios

## 6. Data Persistence

### Response Storage

Each model response is saved to disk:

```python
def _save_response_to_file(self, model_name: str, response: str) -> None:
    # Creates timestamped filename
    # Saves to ./benchmarks/ directory
    # Includes model name and timestamp metadata
```

### Results Storage

Benchmark results are saved in multiple formats:

#### JSON Results
```python
results_filename = f"benchmarks/enhanced_benchmark_results-{timestamp}.json"
# Contains complete raw data for analysis
```

#### Markdown Report
```python
report_filename = f"benchmarks/enhanced_benchmark_report-{timestamp}.md"
# Contains formatted summary tables and detailed results
```

## 7. Quality Metrics (Future)

The system includes placeholders for quality assessment:

- **Response Quality Score**: TBC (To Be Completed)
- **Relevance Score**: TBC (To Be Completed)

These metrics are designed to be extensible for future quality evaluation methods.

## 8. Error Handling

### Graceful Degradation

The system handles failures gracefully:

1. **Tokenizer Failures**: Falls back to simpler counting methods
2. **Model Loading Failures**: Records error and continues with other models
3. **System Monitoring Failures**: Continues with partial metrics
4. **Network Timeouts**: Implements 5-minute timeout for requests

### Error Reporting

Failed runs are tracked with:
- Error messages
- Success/failure rates
- Partial metrics when available

## 9. Configuration

### Benchmark Parameters

Configurable parameters include:

- **Runs per prompt**: Default 3 runs per prompt
- **Timeout**: 5 minutes per request
- **Monitoring frequency**: 100ms sampling rate
- **Base URL**: Ollama API endpoint (default: localhost:11434)

### Model Configuration

Models are defined in the `ModelBenchmark` class:

```python
self.models = {
    "smollm2:1.7b": "smollm2:1.7b",
    "deepseek-r1:1.5b": "deepseek-r1:1.5b", 
    "phi3:mini": "phi3:mini",  
    "gemma:2b": "gemma:2b",
    "llama3.2": "llama3.2",
    "qwen2.5:3b": "qwen2.5:3b",
}
```

## 10. Dependencies

### Required Libraries

- **psutil**: System resource monitoring
- **requests**: HTTP API communication
- **tiktoken**: OpenAI token counting
- **transformers**: HuggingFace tokenizers
- **statistics**: Statistical calculations
- **threading**: Concurrent monitoring
- **subprocess**: Command execution (ollama commands)

### External Dependencies

- **Ollama**: Local model serving
- **Ollama CLI**: Model information and process monitoring

This comprehensive metrics collection system provides detailed insights into LLM performance, system resource utilization, and model characteristics, enabling thorough benchmarking and comparison of different language models. 