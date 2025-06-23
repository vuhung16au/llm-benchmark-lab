# llm-benchmark-lab

This repository provides an advanced benchmarking tool for evaluating large language models (LLMs) running on Ollama. It measures model performance (tokens per second), system resource usage (CPU, memory, GPU), and generates detailed reports. Ideal for comparing LLMs on your local machine with reproducible metrics and automated reporting.

# Enhanced Model Benchmark Guide

This guide explains how to set up your environment and run the `benchmark_models.py` script to benchmark local LLMs using Ollama.

---

## 1. Python Environment Setup

### a. Install Python 3.9 or newer
Ensure you have Python 3.9 or above installed. You can check your version with:

```bash
python3 --version
```

Tested with Python 3.9 and 3.13 under macOS Silicon. s

If you need to install Python, download it from [python.org](https://www.python.org/downloads/) or use your OS package manager.

### b. Create a Virtual Environment
It is recommended to use a virtual environment named `.venv`:

```bash
python3.9 -m venv .venv
```

### c. Activate the Virtual Environment
- On macOS/Linux:
  ```bash
  source .venv/bin/activate
  ```
- On Windows:
  ```cmd
  .venv\Scripts\activate
  ```

### d. Install Python Requirements
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

---

## 2. Ollama Setup

### a. Install Ollama
Follow the instructions at [ollama.com/download](https://ollama.com/download) to install Ollama for your platform.

### b. Start the Ollama Service
Make sure the Ollama service is running:

```bash
ollama serve
```

### c. Pull Required Models
The benchmark script expects the following models to be available locally. Pull them using:

```bash
ollama pull smollm2:1.7b
ollama pull deepseek-r1:1.5b
ollama pull phi3:mini
ollama pull gemma:2b
ollama pull llama3.2
ollama pull qwen2.5:3b
```

Download more models as needed, but ensure the ones listed above are available for the benchmark.


---

## 3. Running the Benchmark Script

Once your environment is set up and models are pulled, run the benchmark script:

```bash
python benchmark_models_enhanced.py
```

The script will:
- Benchmark each model on a test prompt
- Measure tokens per second, time to first token, memory, CPU, and GPU usage
- Save detailed results and a markdown report with a timestamp

Output files will be saved in the current directory and the `benchmarks/` folder.

---

## 4. Output
- **Detailed results:** `enhanced_benchmark_results-<timestamp>.json`
- **Markdown report:** `enhanced_benchmark_report-<timestamp>.md`
- **Model responses:** Saved in the `benchmarks/` directory

---

## 5. Troubleshooting
- Ensure Ollama is running and all required models are pulled before starting the benchmark.
- If you encounter missing package errors, double-check your virtual environment and requirements installation.
- For model-specific issues, verify the model names and tags with `ollama list`.

---

## 6. How to Add a New Ollama Model

To add a new model to the benchmark (e.g., `deepseek-r1:8b`), follow these steps:

### Step 1: Pull the Model from Ollama

First, ensure the model is available locally by pulling it:

```bash
ollama pull deepseek-r1:8b
```

Verify the model is downloaded correctly:

```bash
ollama list
```

### Step 2: Update the Benchmark Script

Edit the `benchmark_models.py` file to include the new model:

#### a. Add to the models dictionary

Find the `self.models` dictionary (around line 272) and add your new model:

```python
self.models = {
    "smollm2:1.7b": "smollm2:1.7b",
    "deepseek-r1:1.5b": "deepseek-r1:1.5b", 
    "deepseek-r1:8b": "deepseek-r1:8b",  # Add this line
    "phi3:mini": "phi3:mini",  
    "gemma:2b": "gemma:2b",
    "llama3.2": "llama3.2",
    "qwen2.5:3b": "qwen2.5:3b",
}
```

#### b. (Optional) Add tokenizer mapping

If you want more accurate token counting, add a tokenizer mapping in the `self.model_tokenizer_map` dictionary (around line 256). For DeepSeek models, the existing "deepseek" mapping should work:

```python
self.model_tokenizer_map = {
    # ...existing mappings...
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-base",  # This covers deepseek-r1 models
    # ...other mappings...
}
```

### Step 3: Test the New Model

Run the benchmark to ensure the new model works correctly:

```bash
python benchmark_models.py
```

The script will now include `deepseek-r1:8b` in the benchmark results.

### Notes

- **Model naming:** Use the exact model name and tag as shown in `ollama list`
- **Performance:** Larger models (like 8B parameters) will take longer to run and use more system resources
- **Tokenizer mapping:** The tokenizer mapping is optional but helps with more accurate token counting for performance metrics
- **Verification:** Always test with a small prompt first to ensure the model responds correctly before running full benchmarks

---

## 7. Troubleshooting

- Ensure Ollama is running and all required models are pulled before starting the benchmark.
- If you encounter missing package errors, double-check your virtual environment and requirements installation.
- For model-specific issues, verify the model names and tags with `ollama list`.


## 8. Configuration Options

### Customizing Benchmark Parameters

You can modify the benchmark behavior by editing these parameters in `benchmark_models_enhanced.py`:

- **Test prompts:** Modify the `self.test_prompts` list to use different evaluation scenarios
- **Sampling parameters:** Adjust `temperature`, `top_p`, `max_tokens` in the model configuration
- **Resource monitoring interval:** Change the monitoring frequency for CPU/memory/GPU metrics
- **Timeout settings:** Modify request timeouts for slower models

## 9. Understanding the Results

### Key Metrics Explained

- **Tokens per second (TPS):** Model generation speed
- **Time to first token (TTFT):** Latency before generation starts
- **Memory usage:** Peak RAM consumption during generation
- **CPU usage:** Average CPU utilization percentage
- **GPU usage:** GPU memory and compute utilization (if available)

### Performance Interpretation

- **High TPS:** Better for real-time applications
- **Low TTFT:** Better user experience for interactive use
- **Memory efficiency:** Important for resource-constrained environments


## 10. Hardware Requirements

### Recommended Setup
- **GPU:** Apple Silicon, NVIDIA RTX series, or AMD with ROCm support
- **Storage:** +1GB for the projects and (optional) models

## 11. Contributing

### Reporting Issues
Please include:
- Operating system and version
- Python version
- Ollama version
- Full error messages and logs

### Adding New Features
- Fork the repository
- Create a feature branch
- Add tests for new functionality
- Submit a pull request with detailed description
---

Happy benchmarking!
