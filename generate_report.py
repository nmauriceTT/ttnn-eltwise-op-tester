#!/usr/bin/env python3
"""
Script to create a PDF report from existing accuracy plots.

This script:
1. Reads operations from report-params.json
2. Creates a markdown file with plots for operations that have images
3. Handles missing images with error messages
4. Converts the markdown to PDF using pandoc
"""

import csv
import os
from stat import FILE_ATTRIBUTE_SYSTEM
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

from jinja2 import Environment, FileSystemLoader


SUMMARY_ROOT = Path("accuracy_results/results")


def _fmt(value):
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.3g}"
    except ValueError:
        return value


_SPECIAL_ROWS = [
    ("-inf", "value_at_x_neg_inf"),
    ("-NaN", "value_at_x_neg_nan"),
    ("-0",   "value_at_x_neg_zero"),
    ("0",    "value_at_x_0"),
    ("+NaN", "value_at_x_pos_nan"),
    ("+inf", "value_at_x_pos_inf"),
]


def _fmt_special(value):
    if value is None or value == "":
        return "—"
    v = value.strip()
    low = v.lower()
    if low == "nan":
        return "NaN"
    if low == "inf":
        return "+inf"
    if low == "-inf":
        return "-inf"
    if v in ("0.0", "0"):
        return "0"
    if v == "-0.0":
        return "-0"
    try:
        return f"{float(v):.3g}"
    except ValueError:
        return v


def load_special_value_table(kind, operation, dtype):
    path = SUMMARY_ROOT / kind / operation / f"summary[{dtype}].csv"
    if not path.is_file():
        return None

    by_impl = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            by_impl[row["implementation"]] = row

    if not by_impl:
        return None

    sample = next(iter(by_impl.values()))
    # golden first, then remaining impls in file order
    columns = (["golden"] if "golden" in by_impl else []) + [k for k in by_impl if k != "golden"]

    table_rows = []
    for label, col_key in _SPECIAL_ROWS:
        if col_key not in sample:
            table_rows.append({"label": label, "values": {c: "—" for c in columns}})
        else:
            table_rows.append({
                "label": label,
                "values": {c: _fmt_special(by_impl[c].get(col_key, "")) for c in columns},
            })

    return {"columns": columns, "table_rows": table_rows}


def load_summary_rows(kind, operation, dtype):
    path = SUMMARY_ROOT / kind / operation / f"summary[{dtype}].csv"
    if not path.is_file():
        return None
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            impl = row.get("implementation", "")
            if impl == "golden":
                continue
            rows.append({
                "implementation": impl,
                "max_ulp": _fmt(row.get("max_ulp_error")),
                "mean_ulp": _fmt(row.get("mean_ulp_error")),
            })
    return rows


def create_markdown_report_jinja2(output_file, dtypes, operations, jinja_template):

    env = Environment(
        loader=FileSystemLoader("templates"),
        autoescape=True
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    unary_operations = operations["unary"]
    binary_operations = operations["binary"]
    unary_bw_operations = operations["unary_bw"]

    unary_summaries = {
        dtype: {op: load_summary_rows("unary", op, dtype) for op in unary_operations}
        for dtype in dtypes
    }

    unary_special_values = {
        dtype: {op: load_special_value_table("unary", op, dtype) for op in unary_operations}
        for dtype in dtypes
    }

    template = env.get_template(jinja_template)

    with open(output_file, "w") as f:
        f.write(template.render(
            unary_operations=unary_operations,
            binary_operations=binary_operations,
            unary_bw_operations=unary_bw_operations,
            unary_summaries=unary_summaries,
            unary_special_values=unary_special_values,
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
    
    
    all_dtypes = ["bfloat16", "float32"]
    
    all_unary_operations = [
        "abs", "identity", "fill", "exp", "exp2", "expm1", "log", "log10", "log2", "log1p", "tanh", "cosh", "sinh", "tan", "atan", "cos",
        "sin", "silu", "gelu", "logit", "swish", "mish", "elu", "celu", "sigmoid", "log_sigmoid", "selu", "softplus", "softsign", "tan",
        "atan", "sin", "cos", "sqrt", "relu", "relu_max", "relu_min", "cbrt", "rsqrt", "reciprocal",
        "digamma", "lgamma", "tanhshrink", "erf", "erfinv"
    ]
    all_binary_operations = [
        "add", "multiply", "hypot", "pow", "divide", "div", "div-accurate", "atan2", "rsub"
    ]
    all_unary_bw_operations = [
        "abs_bw", "floor_bw",
        "exp_bw", "exp2_bw", "expm1_bw",
        "log_bw", "log10_bw", "log2_bw", "log1p_bw",
        "sqrt_bw", "rsqrt_bw",
        "sin_bw", "cos_bw", "tan_bw", "asin_bw", "acos_bw", "atan_bw",
        "sinh_bw", "cosh_bw", "tanh_bw", "asinh_bw", "acosh_bw", "atanh_bw",
        "tanhshrink_bw", "hardtanh_bw", "digamma_bw", "lgamma_bw", "erfinv_bw",
        "sigmoid_bw", "silu_bw", "gelu_bw", "celu_bw", "elu_bw", "selu_bw",
        "softplus_bw", "softsign_bw",
    ]
    all_operations = {
        "unary": all_unary_operations,
        "binary": all_binary_operations,
        "unary_bw": all_unary_bw_operations,
    }

    create_markdown_report_jinja2("accuracy_report.md", all_dtypes, all_operations, "report.md.j2")
    

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
