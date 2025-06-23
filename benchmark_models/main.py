from datetime import datetime
import json
from benchmark_models.model_benchmark import ModelBenchmark
from benchmark_models.test_prompts import test_prompts

if __name__ == "__main__":
    benchmark = ModelBenchmark()
    
    print("Starting Enhanced Model Benchmark...")
    print("This will measure tokens per second plus system metrics for each model.")
    
    # Run benchmark with enhanced metrics
    results = benchmark.run_benchmark(test_prompts, runs_per_prompt=3)
    
    # Print summary to console
    benchmark.print_summary(results)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results with timestamp
    results_filename = f"benchmarks/enhanced_benchmark_results-{timestamp}.json"
    with open(results_filename, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate and save report with timestamp
    report = benchmark.generate_report(results)
    report_filename = f"benchmarks/enhanced_benchmark_report-{timestamp}.md"
    with open(report_filename, "w") as f:
        f.write(report)
    
    print("\nEnhanced benchmark complete!")
    print(f"Detailed results saved to: {results_filename}")
    print(f"Report saved to: {report_filename}") 