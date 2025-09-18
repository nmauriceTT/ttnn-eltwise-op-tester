#!/usr/bin/env python3
"""
Script to create a PDF report from existing accuracy plots.

This script:
1. Reads operations from report-params.json
2. Creates a markdown file with plots for operations that have images
3. Handles missing images with error messages
4. Converts the markdown to PDF using pandoc
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime


def load_report_params(params_file="eltwise-accuracy/report-params.json"):
    """Load report parameters from JSON file."""
    try:
        with open(params_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {params_file} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing {params_file}: {e}")
        return None


def check_image_exists(image_path):
    """Check if an image file exists."""
    return os.path.exists(image_path)


def create_markdown_report(report_params, output_file="accuracy_report.md"):
    """Create a markdown report from report parameters."""
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown_content = f"""# TTNN Eltwise Operations Accuracy Report

Generated on: {timestamp}

This report contains accuracy analysis plots for various TTNN eltwise operations compared to PyTorch reference implementations.

## Overview

The following sections contain accuracy plots for different types of operations:

1. **Unary Operations** - Single-input mathematical functions
2. **Binary Operations** - Two-input mathematical functions

## Plots

"""
    
    # Process each group
    for group in report_params.get("groups", []):
        # Process unary operations
        if "unary" in group:
            unary_data = group["unary"]
            default_description = unary_data.get("default_params", {}).get("description", "Unary operation accuracy analysis")
            
            markdown_content += f"\n# Unary Operations\n\n"
            markdown_content += f"{default_description}\n\n"
            
            for operation in unary_data.get("operations", []):
                operation_id = operation.get("id", "unknown")
                operation_title = operation.get("title", operation_id)
                image_path = operation.get("image", "")
                description = operation.get("description", default_description)
                
                if image_path and check_image_exists(image_path):
                    markdown_content += f"## {operation_title}\n\n"
                    markdown_content += f"{description}\n\n"
                    markdown_content += f"![{operation_title}]({image_path})\n\n"
                else:
                    # Print error message in red (using ANSI color codes)
                    print(f"\033[91mError: Image not found for {operation_title} at {image_path}\033[0m")
        
        # Process binary operations
        if "binary" in group:
            binary_data = group["binary"]
            default_description = binary_data.get("default_params", {}).get("description", "Binary operation accuracy analysis")
            
            markdown_content += f"\n# Binary Operations\n\n"
            markdown_content += f"{default_description}\n\n"
            
            for operation in binary_data.get("operations", []):
                operation_id = operation.get("id", "unknown")
                operation_title = operation.get("title", operation_id)
                image_path = operation.get("image", "")
                description = operation.get("description", default_description)
                
                if image_path and check_image_exists(image_path):
                    markdown_content += f"## {operation_title}\n\n"
                    markdown_content += f"{description}\n\n"
                    markdown_content += f"![{operation_title}]({image_path})\n\n"
                else:
                    # Print error message in red (using ANSI color codes)
                    print(f"\033[91mError: Image not found for {operation_title} at {image_path}\033[0m")
    
    # Add footer
    markdown_content += """
## Notes

- All plots show TTNN implementation accuracy compared to PyTorch reference
- ULP (Unit in the Last Place) errors measure precision in terms of floating-point representation
- Relative errors are shown as percentages
- Binary operation heatmaps show ULP errors across different input value combinations

## Data Sources

- Accuracy data: `accuracy_results/results/`
- Plot configurations: `eltwise-accuracy/plot-params.json` and `eltwise-accuracy/binary-plot-params.json`
- Report configuration: `report-params.json`
- Default `memory_config`

## ULP 

- For accurate operations, ULP error should be < 3 on bfloat16 for useful range
- For approximiate operations, ULP error should ideally be < 10 on bfloat16 (but can depend on operation and range)

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
    """Main function to create PDF report from report parameters."""
    
    print("TTNN Eltwise Operations Accuracy Report Generator")
    print("(From Report Parameters)")
    print("=" * 50)
    
    # Step 1: Load report parameters
    report_params = load_report_params()
    if not report_params:
        print("Failed to load report parameters. Exiting.")
        return 1
    
    print("Report parameters loaded successfully")
    
    # Step 2: Create markdown report
    markdown_file = create_markdown_report(report_params)
    
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
