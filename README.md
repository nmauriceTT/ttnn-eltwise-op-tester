# TTNN Eltwise Operation Tester

Test and plot accuracy of TTNN's element-wise operations.

## Setup 

This repository relies on tt-metal configuration.

```bash
PYTHONPATH=<path/to/tt-metal>
TT_METAL_HOME=<path/to/tt-metal>
source <path/to/tt-metal>/python_env/bin/activate
```

## Directory Structure

```
├── configs/
│   ├── unary-plots.json      # Plot configuration for unary operations
│   └── binary-plots.json     # Plot configuration for binary operations
├── accuracy_results/
│   ├── results/
│   │   ├── unary/            # Raw accuracy measurement results (CSV)
│   │   ├── unary-bw/         # Raw accuracy measurement results for backward ops (CSV)
│   │   └── binary/           # Raw accuracy measurement results (CSV)
│   └── plots/
│       ├── unary/            # Generated plots for unary operations
│       ├── unary-bw/         # Generated plots for backward operations
│       └── binary/           # Generated plots for binary operations
├── templates/
│   └── report.md.j2          # Jinja2 template for PDF report
├── measure_accuracy.py       # Main script for accuracy measurements
├── plot.py                   # Plot generation for unary operations
├── plot_binary.py            # Plot generation for binary operations
└── generate_report.py        # PDF report generation
```

## Accuracy Benchmark

The `measure_accuracy.py` script measures accuracy for both unary and binary operations. The operation type is automatically detected.

### Unary Operations

#### bfloat16
```bash
python measure_accuracy.py -t "bfloat16"
```

Test a specific operation:
```bash
python measure_accuracy.py -t "bfloat16" -o "exp"
```

#### float32
> Note: Not optimized, takes ~2 minutes per operation

```bash
python measure_accuracy.py -t "float32"
```

### Binary Operations

```bash
python measure_accuracy.py -t "bfloat16" -o "atan2"
```

### Backward Operations

Test backward (gradient) operations with grad=1. Results go to `accuracy_results/results/unary-bw/`.

```bash
python measure_accuracy.py --backward -t "bfloat16"
```

Test a specific backward operation:
```bash
python measure_accuracy.py --backward -t "bfloat16" -o "exp_bw"
python measure_accuracy.py --backward -t "bfloat16" -o "gelu_bw"
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--type` | `-t` | Data type (`bfloat16` or `float32`) | `bfloat16` |
| `--operation` | `-o` | Specific operation to test | All operations |
| `--output-dir` | `-O` | Output directory for results | `accuracy_results/results/` |
| `--group-size` | `-g` | Measurement batch size (unary only) | 1 (bf16) / 65536 (f32) |
| `--backward` | | Run backward operations instead of forward | `false` |

## Plot Generation

### Unary Operations

```bash
python plot.py
```

Reads configuration from `configs/unary-plots.json` and outputs plots to `accuracy_results/plots/unary/`.

### Binary Operations

```bash
python plot_binary.py
```

Reads configuration from `configs/binary-plots.json` and outputs plots to `accuracy_results/plots/binary/`.

## Performance Benchmarking

The `bench.py` script measures operation throughput via tracy profiling. Results are written to `generated/benchmarks/<type>/`.

See [`PERFORMANCE_MICROBENCHMARK.md`](PERFORMANCE_MICROBENCHMARK.md) for details on what is measured (tensor layout, warmup, supported op types).

### Run all unary operations

```bash
python bench.py --type unary -t bfloat16
```

### Run all binary operations

```bash
python bench.py --type binary -t bfloat16
```

### Run a specific operation (all variants)

```bash
python bench.py -k exp -t bfloat16
```

### Disassemble kernels (`--dump-asm`)

Pass `--dump-asm` to write each tested kernel's MATH-thread (trisc1) disassembly and an SFPU instruction histogram to `generated/benchmarks/<type>/asm/`. Each run uses an isolated kernel cache so the compute kernel ELF is unambiguous.

```bash
python bench.py -k abs -t bfloat16 --dump-asm
```

Output files are named `<variant>_<dtype>_trisc1.asm` (e.g. `abs_bfloat16_trisc1.asm`). When an op has multiple implementation variants, the base name is included (e.g. `exp_exp-fast-approx_bfloat16_trisc1.asm`).

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--type` | | Operation type (`unary` or `binary`) | `unary` |
| `--dtype` | `-t` | Data type (`bfloat16` or `float32`) | `bfloat16` |
| `--operation` | `-k` | Filter by base operation name (runs all variants) | All operations |
| `--dump-asm` | | Disassemble each kernel's trisc1 ELF to `<output>/asm/` | off |

### Output Files

Results are saved as CSV files in `generated/benchmarks/<type>/`:
- `<op>.csv` — results for a specific operation (when `-k` is used)
- `processed_results.csv` — results for all operations
- `asm/<variant>_<dtype>_trisc1.asm` — kernel disassembly (when `--dump-asm` is used)

Each CSV contains `implementation_name`, `cycles_per_datum`, and `cycles_per_tile` columns.

## Accuracy Report Generation

Generate a comprehensive PDF report with all accuracy plots.

### Prerequisites

#### Python Packages

```bash
pip install matplotlib seaborn pandas numpy jinja2 loguru scipy
```

#### PDF Generation Tools

The report generator uses **pandoc** with **pdflatex** as the PDF engine.

**Ubuntu/Debian:**
```bash
sudo apt-get install pandoc texlive-latex-recommended texlive-fonts-recommended
```




### Usage

```bash
python generate_report.py
```

### Output Files

- `accuracy_report.md` - Markdown report with all plots
- `accuracy_report.pdf` - PDF report (if pandoc/pdflatex are available)

## Troubleshooting

### No plots found
Ensure:
1. Accuracy data exists in `accuracy_results/results/`
2. Plot configuration files are present in `configs/`
3. Run the plot generation scripts first (`plot.py`, `plot_binary.py`)

### PDF conversion fails
1. Verify pandoc and LaTeX (pdflatex) are installed
2. Check that the markdown file was created
3. Manual conversion: `pandoc accuracy_report.md -o accuracy_report.pdf --pdf-engine=pdflatex`

### Plot generation fails
1. Check that all required Python packages are installed
2. Verify that accuracy data files exist in `accuracy_results/results/`
3. Check plot configuration files in `configs/` for syntax errors

## Example Workflow

```bash
# 1. Ensure you're in the project directory
cd /path/to/ttnn-eltwise-op-tester

# 2. Set up tt-metal environment
source <path/to/tt-metal>/python_env/bin/activate
export PYTHONPATH=<path/to/tt-metal>
export TT_METAL_HOME=<path/to/tt-metal>

# 3. Install additional dependencies (if needed)
pip install matplotlib seaborn pandas numpy jinja2 loguru scipy

# 4. Run accuracy measurements
python measure_accuracy.py -t "bfloat16"

# 5. Generate plots
python plot.py
python plot_binary.py

# 6. Generate PDF report
python generate_report.py

# 7. View the results
ls -la accuracy_report.*
```

## Notes

- Plots are organized by error type (ULP, relative, absolute, value)
- Plots are sorted alphabetically for consistent ordering
- The PDF report includes a table of contents for easy navigation
- All plots are generated with descriptive titles based on operation names
