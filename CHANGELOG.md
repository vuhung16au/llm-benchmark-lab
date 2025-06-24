# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Documentation improvements (architecture.md, METRICS.md, etc.)
- Planning for future extensibility and quality metrics

## [0.2.0] - 2025-06-23
### Added
- Modularized codebase into `benchmark_models/` package
- System resource monitoring (CPU, memory, GPU) via `SystemMonitor`
- Accurate token counting with HuggingFace and tiktoken (`TokenCounter`)
- Streaming benchmark logic and timing (tokens/sec, time to first token)
- Model size and parameter extraction from Ollama
- Markdown and JSON report generation
- Output of all model responses to `benchmarks/` directory
- Extensible prompt and model configuration
- Enhanced README and setup instructions

### Changed
- Refactored code for maintainability and extensibility
- Improved error handling and fallback logic for token counting

### Fixed
- Minor bug fixes in metrics collection and reporting

### Breaking Changes
- The main script entrypoint is now `benchmark_models/main.py` (not `benchmark_models.py`)
- All model and prompt configuration is now in the `benchmark_models/` package

## [0.1.0] - 2025-06-23
### Added
- Initial commit: basic benchmarking script for LLMs on Ollama
- Simple performance measurement (tokens/sec)
- Basic prompt and model configuration 