# Folder Structure

This document describes the folder structure and organization of the llm-benchmark-lab project.

## Project Root Directory

```text
llm-benchmark-lab/
├── .git/                                          # Git repository metadata
├── .gitignore                                     # Git ignore configuration
├── .venv/                                         # Python virtual environment
├── __pycache__/                                   # Python bytecode cache
│   └── benchmark_models.cpython-313.pyc            # Compiled Python bytecode
├── benchmarks/                                    # Model response outputs
│   ├── response-deepseek-r1-1.5b-*.txt             # DeepSeek R1 1.5B model responses
│   ├── response-phi3-mini-*.txt                    # Phi3 Mini model responses
│   └── response-smollm2-1.7b-*.txt                 # SmolLM2 1.7B model responses
├── benchmark_models/                              # Benchmarking package (modularized)
│   ├── __init__.py                                # Package marker
│   ├── main.py                                    # Entrypoint script (run this)
│   ├── model_benchmark.py                         # ModelBenchmark class (benchmark logic)
│   ├── system_monitor.py                          # SystemMonitor class (resource monitoring)
│   ├── token_counter.py                           # TokenCounter class (token counting)
│   └── test_prompts.py                            # List of test prompts
├── Benchmark-Report-Checkpoint-20250623-1400.md    # Benchmark report checkpoint
├── FOLDER-STRUCTURE.md                            # This file
├── LICENSE                                        # Project license
├── README.md                                      # Project documentation
└── requirements.txt                               # Python dependencies
```

## Directory Descriptions

### Root Files

| File | Description |
|------|-------------|
| `benchmark_models.py` | The main Python script that runs LLM benchmarks using Ollama. Contains all benchmarking logic, performance measurement, and report generation. |
| `requirements.txt` | Lists all Python package dependencies required to run the benchmark script. |
| `README.md` | Comprehensive project documentation including setup instructions, usage guide, and troubleshooting tips. |
| `LICENSE` | Software license for the project. |
| `Benchmark-Report-Checkpoint-*.md` | Generated benchmark reports with performance metrics and analysis. |

### Directories

#### `.git/`

Contains Git version control metadata and history. This is automatically created when the repository is initialized and should not be modified manually.

#### `.venv/`

Python virtual environment directory. Contains:

- Isolated Python installation
- Project-specific package installations
- Virtual environment activation scripts

This directory is typically excluded from version control and should be recreated on each system using the setup instructions in README.md.

#### `__pycache__/`

Python automatically generates this directory to store compiled bytecode (.pyc files) for faster script execution. Contents:

- `benchmark_models.cpython-313.pyc` - Compiled bytecode for the main script

This directory is automatically managed by Python and can be safely deleted.

#### `benchmarks/`

Storage location for all model response outputs generated during benchmark runs. Files are organized by:

- **Model name**: Identifies which LLM generated the response
- **Timestamp**: Shows when the benchmark was executed
- **Format**: Plain text files containing the raw model responses

Example file naming pattern:

```text
response-{model-name}-{timestamp}.txt
```

Current models being benchmarked:

- `deepseek-r1:1.5b` - DeepSeek R1 1.5B parameter model
- `phi3:mini` - Microsoft Phi3 Mini model  
- `smollm2:1.7b` - SmolLM2 1.7B parameter model

### benchmark_models/

| File | Description |
|------|-------------|
| `__init__.py` | Package marker for Python imports |
| `main.py` | Entrypoint script. Run this to start the benchmark |
| `model_benchmark.py` | Contains the ModelBenchmark class and core benchmarking/report logic |
| `system_monitor.py` | Contains the SystemMonitor class for resource monitoring |
| `token_counter.py` | Contains the TokenCounter class for token counting |
| `test_prompts.py` | Contains the test prompts used for benchmarking |

## File Naming Conventions

### Benchmark Response Files

- Format: `response-{model-name}-{timestamp}.txt`
- Timestamp format: `YYYYMMDD_HHMMSS`
- Model names use lowercase with hyphens instead of colons for filesystem compatibility

### Report Files

- Format: `Benchmark-Report-Checkpoint-{YYYYMMDD-HHMM}.md`
- Contains comprehensive analysis and metrics from benchmark runs

## Adding New Content

### New Models

When adding new models to benchmark:

1. Response files will automatically be created in `benchmarks/`
2. Follow the existing naming convention
3. Update this documentation if new model families are added

### New Output Types

If extending the benchmark to generate additional output types:

1. Consider creating subdirectories within `benchmarks/` for organization
2. Update this documentation to reflect new structure
3. Maintain consistent naming conventions

## Maintenance Notes

- The `__pycache__/` directory can be safely deleted and will be regenerated automatically
- The `.venv/` directory should be recreated rather than copied between systems
- Benchmark response files in `benchmarks/` can grow large over time and may need periodic cleanup
- Generated report files contain valuable historical data and should be preserved

## File Size Considerations

- Model response files are typically small (< 1KB each)
- The `benchmarks/` directory will grow with each benchmark run
- Virtual environment (`.venv/`) is typically 50-200MB depending on installed packages
- Git repository (`.git/`) size depends on project history and stored files
