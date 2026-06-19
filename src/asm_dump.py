import glob
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


def implementation_label(base_operation_name, implementation_name, suffix=None):
    """Stable label for an op variant; omit base name when it matches the implementation."""
    if base_operation_name == implementation_name:
        label = base_operation_name
    else:
        label = f"{base_operation_name}_{implementation_name}"
    if suffix is not None:
        label = f"{label}_{suffix}"
    return label


def _find_compute_kernel_elf(cache_root):
    """Newest compute-kernel trisc1 (MATH/SFPU thread) ELF under an isolated cache.

    Recursive because TT_METAL_CACHE glues the build_key onto the dir name
    (tt-metal-cache<key>/kernels/...), unlike the default cache's extra level.
    trisc1 only exists for compute kernels; cq_* are dispatch infra, not the op
    under test. Custom sfpi/TTI kernels land under kernels/Kernel_Source_Code/,
    but matching newest-non-cq also covers stock ops (kernels/eltwise_sfpu/).
    """
    elfs = glob.glob(os.path.join(cache_root, "**", "kernels", "*", "*", "trisc1", "trisc1.elf"), recursive=True)
    elfs = [p for p in elfs if not Path(p).parents[2].name.startswith("cq_")]
    if not elfs:
        return None
    return max(elfs, key=os.path.getmtime)


def dump_kernel_asm(cache_root, asm_out_dir, label):
    """Disassemble the op's MATH-thread kernel and write it + an SFPU histogram.

    Returns a dict (label, asm_path, sfp_total, histogram) or None on failure.
    """
    metal_home = os.getenv("TT_METAL_HOME")
    if not metal_home:
        print("asm-dump: TT_METAL_HOME is not set; skipping")
        return None
    objdump = os.path.join(metal_home, "runtime", "sfpi", "compiler", "bin", "riscv-tt-elf-objdump")
    if not os.path.isfile(objdump):
        print(f"asm-dump: objdump not found at {objdump}; skipping")
        return None

    elf = _find_compute_kernel_elf(cache_root)
    if elf is None:
        print(f"asm-dump: no compute-kernel trisc1.elf under {cache_root}; skipping")
        return None

    try:
        disasm = subprocess.run([objdump, "-d", elf], check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e:
        print(f"asm-dump: objdump failed for {elf}: {e.stderr}")
        return None

    os.makedirs(asm_out_dir, exist_ok=True)
    asm_path = os.path.join(asm_out_dir, f"{label}_trisc1.asm")
    with open(asm_path, "w") as f:
        f.write(disasm)

    # objdump -d formats each instruction as "addr:\tbytes\tmnemonic operands".
    hist = Counter()
    for line in disasm.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2].split():
            mnem = parts[2].split()[0]
            if "sfp" in mnem.lower():
                hist[mnem] += 1

    total = sum(hist.values())
    print(f"asm-dump: {label}: {total} SFPU instrs -> {asm_path}")
    for mnem, count in hist.most_common():
        print(f"    {count:3d}  {mnem}")
    return {"label": label, "asm_path": asm_path, "sfp_total": total, "histogram": dict(hist)}


def dump_implementation_asm(implementation_name, base_operation_name, dtype, asm_out_dir, operation_type="unary"):
    """Run an op once in an isolated kernel cache, then disassemble its trisc1 ELF."""
    if not asm_out_dir:
        raise ValueError("asm_out_dir must be set")
    label = implementation_label(base_operation_name, implementation_name, dtype)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_op_script = os.path.join(project_root, "templates", "run-op.py")

    # Isolated, empty kernel cache: forces a fresh compile and makes the op's
    # compute kernel the only non-dispatch trisc1.elf, so the ELF is
    # unambiguous. TT_METAL_CACHE has "tt-metal-cache" appended (rtoptions.cpp).
    asm_cache_base = tempfile.mkdtemp(
        prefix=f"asmcache_{implementation_label(base_operation_name, implementation_name)}_"
    )
    env = os.environ.copy()
    env["TT_METAL_CACHE"] = asm_cache_base
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (project_root, env.get("PYTHONPATH", "")) if p
    )

    cmd = [sys.executable, run_op_script, operation_type, dtype, implementation_name]
    print(f"Running op: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, cwd=project_root)
        return dump_kernel_asm(asm_cache_base, asm_out_dir, label)
    except subprocess.CalledProcessError as e:
        print(f"asm-dump: op run failed for {implementation_name}: {e}")
        if e.stderr:
            print(e.stderr)
        if e.stdout:
            print(e.stdout)
        return None
    finally:
        shutil.rmtree(asm_cache_base, ignore_errors=True)
