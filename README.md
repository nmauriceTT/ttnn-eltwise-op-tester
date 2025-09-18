# Setup 
For now, this repository relies on tt-metal configuration.

```
PYTHONPATH=<path/to/tt-metal>
TT_METAL_HOME=<path/to/tt-metal>
source <path/to/tt-metal>/python_env/bin/activate
```

# Accuracy Benchmark

For unary operations:
```
python eltwise-accuracy/measure_accuracy.py
```

For binary operations:
```
python eltwise-accuracy/measure-binary.py
```

# Plot generation

For unary operations:
```
python eltwise-accuracy/plot_accuracy.py
```


For binary operations:
```
python eltwise-accuracy/plot_binary.py
```

# Accuracy Report Generation

This directory contains scripts to generate comprehensive PDF reports of TTNN eltwise operation accuracy analysis.

## Scripts

### 1. `generate_accuracy_report.py`
**Full workflow script** that:
- Generates all accuracy plots from scratch
- Creates a markdown report with all plots
- Converts the markdown to PDF using pandoc

**Usage:**
```bash
python3 generate_accuracy_report.py
```

### 2. `generate_report_from_existing_plots.py`
**Simpler script** that:
- Uses existing plot files (if any)
- Creates a markdown report with existing plots
- Converts the markdown to PDF using pandoc

**Usage:**
```bash
python3 generate_report_from_existing_plots.py
```

## Prerequisites

### Required Software
1. **Python 3** with required packages:
   - matplotlib
   - seaborn
   - pandas
   - numpy

2. **pandoc** for PDF conversion:
   - Ubuntu/Debian: `sudo apt-get install pandoc texlive-latex-recommended`
   - macOS: `brew install pandoc`
   - Windows: Download from https://pandoc.org/installing.html

3. **LaTeX** (for PDF generation):
   - Ubuntu/Debian: `sudo apt-get install texlive-latex-recommended`
   - macOS: `brew install --cask mactex`
   - Windows: Install MiKTeX or TeX Live

### Data Requirements
- Accuracy data files in `accuracy_results/results/`
- Plot configuration files:
  - `eltwise-accuracy/plot-params.json`
  - `eltwise-accuracy/binary-plot-params.json`

## How to Use

```bash
python3 eltwise-accuracy/generate_report.py
```

### Output Files

- **`accuracy_report.md`** - Markdown report with all plots
- **`accuracy_report.pdf`** - PDF report (if pandoc is available)


## Troubleshooting

### No plots found
If you get "No plot files found", ensure:
1. Accuracy data exists in `accuracy_results/results/`
2. Plot configuration files are present
3. Run the plot generation scripts first

### PDF conversion fails
If pandoc fails:
1. Install pandoc and LaTeX
2. Check that the markdown file was created
3. Manually convert: `pandoc accuracy_report.md -o accuracy_report.pdf`

### Plot generation fails
If plot generation fails:
1. Check that all required Python packages are installed
2. Verify that accuracy data files exist
3. Check plot configuration files for syntax errors

## Example Workflow

```bash
# 1. Ensure you're in the project directory
cd /path/to/ttnn-eltwise-op-tester

# 2. Install dependencies (if needed)
pip install matplotlib seaborn pandas numpy

# 3. Install pandoc and LaTeX (if needed)
sudo apt-get install pandoc texlive-latex-recommended

# 4. Generate the complete report
python3 eltwise-accuracy/generate_report.py

# 5. View the results
ls -la accuracy_report.*
```

## Notes

- The scripts automatically organize plots by type (ULP, relative, absolute, etc.)
- Plots are sorted alphabetically for consistent ordering
- The PDF includes a table of contents for easy navigation
- All plots are included with descriptive titles based on filenames
