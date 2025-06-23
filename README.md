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

Happy benchmarking! 