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
│   │   └── binary/           # Raw accuracy measurement results (CSV)
│   └── plots/
│       ├── unary/            # Generated plots for unary operations
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

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--type` | `-t` | Data type (`bfloat16` or `float32`) | `bfloat16` |
| `--operation` | `-o` | Specific operation to test | All operations |
| `--output-dir` | `-O` | Output directory for results | `accuracy_results/results/` |
| `--group-size` | `-g` | Measurement batch size (unary only) | 1 (bf16) / 65536 (f32) |

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
