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
from stat import FILE_ATTRIBUTE_SYSTEM
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

from jinja2 import Environment, FileSystemLoader


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

def create_markdown_report_jinja2(output_file, dtypes, operations, jinja_template):

    env = Environment(
        loader=FileSystemLoader("templates"),
        autoescape=True
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    unary_operations = operations["unary"]
    binary_operations = operations["binary"]

    template = env.get_template(jinja_template)

    with open(output_file, "w") as f:
        f.write(template.render(
            unary_operations=unary_operations, 
            binary_operations=binary_operations, 
            timestamp=timestamp, 
            dtypes=dtypes
        ))


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
    
    all_dtypes = ["bfloat16", "float32"]
    
    all_unary_operations = [
        "exp", "exp2", "log", "log10", "log2,", "log1p", "tanh", "cosh", "sinh", "tan", "atan", "cos", 
        "sin", "silu", "gelu", "logit", "swish", "mish", "elu", "selu", "softplus", "softsign", "tan", 
        "atan2", "sin", "cos", "sqrt", "cbrt", "cbrt-pow1d3", "cbrt-pow1d3-fp32", "rsqrt", "rsqrt_approx",
        "reciprocal", "digamma", "lgamma", "tanhshrink"
    ]
    all_binary_operations = [
        "divide", "div", "div-accurate", "pow"
    ]
    all_operations = {
        "unary": all_unary_operations,
        "binary": all_binary_operations
    }

    print("Report parameters loaded successfully")

    # Step 2: Create markdown report
    create_markdown_report_jinja2("accuracy_report.md", all_dtypes, all_operations, "report.md.j2")
    # markdown_file = create_markdown_report(report_params)
    
    # Step 3: Convert to PDF
    if convert_to_pdf("accuracy_report.md"):
        print("\n" + "=" * 50)
        print("SUCCESS: Accuracy report generated successfully!")
        print(f"Markdown file: accuracy_report.md")
        print("PDF file: accuracy_report.pdf")
        print("=" * 50)
        return 0
    else:
        print("\n" + "=" * 50)
        print("WARNING: Markdown report created but PDF conversion failed.")
        print(f"Markdown file: accuracy_report.md")
        print("You can manually convert it to PDF using pandoc or view it in a markdown viewer.")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())
