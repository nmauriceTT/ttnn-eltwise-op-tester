#!/usr/bin/env python3
"""
Script to create a PDF report from existing accuracy plots.

This script:
1. Finds all existing plot files in the accuracy_results/plots directory
2. Creates a markdown file with all plots included
3. Converts the markdown to PDF using pandoc
"""

import os
import sys
import subprocess
import glob
from pathlib import Path
from datetime import datetime


def find_all_plot_files():
    """Find all existing plot files."""
    plot_files = []
    
    # Look for PNG files in the accuracy_results/plots directory and subdirectories
    plot_dirs = [
        "accuracy_results/plots",
        "accuracy_results/plots/ulp",
        "accuracy_results/plots/abs", 
        "accuracy_results/plots/value",
        "accuracy_results/plots/binary_heatmaps"
    ]
    
    for plot_dir in plot_dirs:
        if os.path.exists(plot_dir):
            png_files = glob.glob(os.path.join(plot_dir, "*.png"))
            plot_files.extend(png_files)
    
    # Sort files for consistent ordering
    plot_files.sort()
    
    return plot_files


def create_markdown_report(plot_files, output_file="accuracy_report.md"):
    """Create a markdown report with all plots."""
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown_content = f"""# TTNN Eltwise Operations Accuracy Report

Generated on: {timestamp}

This report contains accuracy analysis plots for various TTNN eltwise operations compared to PyTorch reference implementations.

## Overview

The following sections contain accuracy plots for different types of operations:

1. **ULP Error Plots** - Show Unit in the Last Place (ULP) errors
2. **Relative Error Plots** - Show relative percentage errors  
3. **Absolute Error Plots** - Show absolute errors
4. **Value Comparison Plots** - Show actual function values
5. **Binary Operation Heatmaps** - Show ULP errors for binary operations

## Plots

"""
    
    # Group plots by type based on directory structure
    plot_groups = {
        "ULP Error Plots": [],
        "Relative Error Plots": [],
        "Absolute Error Plots": [],
        "Value Comparison Plots": [],
        "Binary Operation Heatmaps": []
    }
    
    for plot_file in plot_files:
        plot_name = os.path.basename(plot_file)
        relative_path = os.path.relpath(plot_file, ".")
        
        if "/ulp/" in plot_file:
            plot_groups["ULP Error Plots"].append((plot_name, relative_path))
        # elif "/abs/" in plot_file:
        #     plot_groups["Absolute Error Plots"].append((plot_name, relative_path))
        # elif "/value/" in plot_file:
        #     plot_groups["Value Comparison Plots"].append((plot_name, relative_path))
        # elif "/binary_heatmaps/" in plot_file:
        #     plot_groups["Binary Operation Heatmaps"].append((plot_name, relative_path))
        # else:
        #     # Default to relative error plots for files in the main plots directory
        #     plot_groups["Relative Error Plots"].append((plot_name, relative_path))
    
    # Add each group to the markdown
    for group_name, plots in plot_groups.items():
        if plots:
            markdown_content += f"\n# {group_name}\n\n"
            
            for plot_name, relative_path in plots:
                # Extract operation name from filename for better titles
                operation_name = plot_name.replace("-bfloat16.png", "").replace("-bfloat16-zoom.png", " (zoomed)").replace("-bfloat16-ulp.png", "").replace("-bfloat16-abs.png", "").replace("-bfloat16-value.png", "")
                operation_name = operation_name.replace("-", " ").title()
                
                markdown_content += f"## {operation_name}\n\n"
                markdown_content += f"![{operation_name}]({relative_path})\n\n"
    
    # Add footer
    markdown_content += """
## Notes

- All plots show TTNN implementation accuracy compared to PyTorch reference
- ULP (Unit in the Last Place) errors measure precision in terms of floating-point representation
- Relative errors are shown as percentages
- Binary operation heatmaps show ULP errors across different input value combinations
- Plots are generated using matplotlib and seaborn

## Data Sources

- Accuracy data: `accuracy_results/results/`
- Plot configurations: `eltwise-accuracy/plot-params.json` and `eltwise-accuracy/binary-plot-params.json`
"""
    
    # Write markdown file
    with open(output_file, 'w') as f:
        f.write(markdown_content)
    
    print(f"Markdown report created: {output_file}")
    return output_file


def convert_to_pdf(markdown_file, output_pdf="accuracy_report.pdf"):
    """Convert markdown file to PDF using pandoc."""
    
    print(f"Converting {markdown_file} to PDF...")
    
    try:
        # Check if pandoc is available
        subprocess.run(['pandoc', '--version'], check=True, capture_output=True)
        
        # Convert markdown to PDF
        cmd = [
            'pandoc',
            markdown_file,
            '-f', 'markdown-implicit_figures',
            '-o', output_pdf,
            '--pdf-engine=pdflatex',
            '--variable', 'geometry:margin=1in',
            '--variable', 'fontsize=10pt',
            '--variable', 'documentclass=article',
            '--toc',  # Add table of contents
            '--toc-depth=3'
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"PDF report created: {output_pdf}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting to PDF: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: pandoc not found. Please install pandoc to generate PDF.")
        print("On Ubuntu/Debian: sudo apt-get install pandoc texlive-latex-recommended")
        print("On macOS: brew install pandoc")
        return False


def main():
    """Main function to create PDF report from existing plots."""
    
    print("TTNN Eltwise Operations Accuracy Report Generator")
    print("(From Existing Plots)")
    print("=" * 50)
    
    # Step 1: Find all existing plot files
    plot_files = find_all_plot_files()
    print(f"Found {len(plot_files)} plot files")
    
    if not plot_files:
        print("No plot files found in accuracy_results/plots/")
        print("Please run the plot generation scripts first:")
        print("  cd eltwise-accuracy")
        print("  python plot_accuracy.py")
        print("  python plot_binary.py")
        return 1
    
    # Step 2: Create markdown report
    markdown_file = create_markdown_report(plot_files)
    
    # Step 3: Convert to PDF
    if convert_to_pdf(markdown_file):
        print("\n" + "=" * 50)
        print("SUCCESS: Accuracy report generated successfully!")
        print(f"Markdown file: {markdown_file}")
        print("PDF file: accuracy_report.pdf")
        print("=" * 50)
        return 0
    else:
        print("\n" + "=" * 50)
        print("WARNING: Markdown report created but PDF conversion failed.")
        print(f"Markdown file: {markdown_file}")
        print("You can manually convert it to PDF using pandoc or view it in a markdown viewer.")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())
