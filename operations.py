import ttnn
import torch
import math
import numpy as np
from utils import TERM_RED, TERM_RESET
import struct



from kernel_generator import generate_unary_kernel_from_polynomial, generate_unary_kernel_from_sfpi_source, generic_unary_kernel, generate_kernel_source_code_from_llk


global_device = None

def run_op_fp32(ttnn_op, args, device=None):
    ttnn_args = [convert_to_ttn(arg, device) for arg in args]
    return ttnn_op(*ttnn_args)


def convert_to_ttn(tensor, device):

    if isinstance(tensor, ttnn.Tensor):
        return tensor

    # Shard data on all cores to speed-up computation
    ttnn_tensor = ttnn.from_torch(tensor, layout=ttnn.Layout.TILE, device=device)
    
    return ttnn_tensor

def run_ttnn_fp32_and_round_bf16(ttnn_op, args):

    global global_device
    device = args[0].device
    assert device is not None

    host_bf16_args = [convert_to_ttn(arg, device) for arg in args]
    host_f32_args = [arg.to(torch.float32) for arg in host_bf16_args]
    ttnn_args = [ttnn.from_torch(arg, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device) for arg in host_f32_args]

    result = ttnn_op(*ttnn_args)
    
    # Convert back to bf16
    result = ttnn.to_torch(result)
    result = result.to(torch.bfloat16)
    result = ttnn.from_torch(result, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return result


def exp_fp32(x):
    """
    Compute exp(x) using only fp32 internal arithmetic.
    Vectorized to handle arrays element-wise.

    Algorithm:
    1. Handle special cases (overflow, underflow)
    2. Convert to base-2: exp(x) = 2^(x/ln2)
    3. Range reduction: z = x/ln2 = k + r where k is integer, r in [-0.5, 0.5]
    4. Compute 2^r using polynomial approximation
    5. Scale by 2^k: result = 2^k * 2^r

    IMPORTANT: All intermediate computations use ONLY fp32 precision.
    """

    ttnn_dtype = x.dtype
    x = ttnn.to_torch(x, dtype=torch.float32)
    x = x.numpy()

    x = np.float32(x)

    # Constants (all in fp32)
    INV_LN2 = np.float32(1.4426950408889634)  # 1/ln(2)
    OVERFLOW_THRESHOLD = np.float32(88.0)
    UNDERFLOW_THRESHOLD = np.float32(-103.97)
    HALF = np.float32(0.5)
    ONE = np.float32(1.0)

    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=np.float32)

    # Handle special cases (vectorized)
    nan_mask = np.isnan(x)
    overflow_mask = x >= OVERFLOW_THRESHOLD
    underflow_mask = x <= UNDERFLOW_THRESHOLD
    
    # Set special case values
    result[nan_mask] = np.float32(np.nan)
    result[overflow_mask] = np.float32(np.inf)
    result[underflow_mask] = np.float32(0.0)
    
    # Mask for normal computation (not special cases)
    normal_mask = ~(nan_mask | overflow_mask | underflow_mask)

    # Only compute for normal values
    if np.any(normal_mask):
        x_normal = x[normal_mask]

        # Step 1: Convert to base-2 exponential
        # z = x / ln(2)
        z = np.float32(x_normal * INV_LN2)

        # Step 2: Range reduction
        # Split z = k + r where k is integer, r in [-0.5, 0.5]
        k_temp = np.float32(z + HALF)
        k = np.float32(np.floor(k_temp))
        k_int = k.astype(np.int32)  # Convert to integer array

        # BUG FIX: Compute r = z - k (NOT r = x - k*ln2)
        # This is correct because we want: 2^z = 2^(k+r) = 2^k * 2^r
        r = np.float32(z - k)

        # Step 3: Polynomial approximation for 2^r
        # Using Taylor series: 2^r = exp(r*ln2) = 1 + r*ln2 + (r*ln2)²/2! + ...
        # For better accuracy, compute s = r*ln2, then use exp(s) polynomial

        LN2 = np.float32(0.6931471805599453)

        # Compute s = r * ln(2), all in fp32
        s = np.float32(r * LN2)

        # Now approximate exp(s) using Taylor series
        # exp(s) = 1 + s + s²/2! + s³/3! + s⁴/4! + s⁵/5! + s⁶/6!
        # Coefficients for exp(s)
        c6 = np.float32(1.0 / 720.0)     # 1/6!
        c5 = np.float32(1.0 / 120.0)     # 1/5!
        c4 = np.float32(1.0 / 24.0)      # 1/4!
        c3 = np.float32(1.0 / 6.0)       # 1/3!
        c2 = np.float32(0.5)             # 1/2!
        c1 = np.float32(1.0)             # 1/1!
        c0 = np.float32(1.0)             # constant

        # Horner's method for better accuracy (vectorized)
        # p = c0 + s*(c1 + s*(c2 + s*(c3 + s*(c4 + s*(c5 + s*c6)))))
        p = np.float32(c6)

        temp = np.float32(p * s)
        p = np.float32(c5 + temp)

        temp = np.float32(p * s)
        p = np.float32(c4 + temp)

        temp = np.float32(p * s)
        p = np.float32(c3 + temp)

        temp = np.float32(p * s)
        p = np.float32(c2 + temp)

        temp = np.float32(p * s)
        p = np.float32(c1 + temp)

        temp = np.float32(p * s)
        p = np.float32(c0 + temp)

        # Step 4: Scale by 2^k using ldexp (fp32 only, vectorized)
        # ldexp is just bit manipulation of exponent, works fine on fp32
        result_normal = np.float32(np.ldexp(p, k_int))
        
        # Assign computed values back to result array
        result[normal_mask] = result_normal

    result = torch.from_numpy(result)
    result = ttnn.from_torch(result, dtype=ttnn_dtype)

    return result

def exp_claude1(x):
    """
    Compute exp(x) using only fp32 internal arithmetic.
    Vectorized to handle arrays element-wise.

    Algorithm:
    1. Handle special cases (overflow, underflow)
    2. Convert to base-2: exp(x) = 2^(x/ln2)
    3. Range reduction: z = x/ln2 = k + r where k is integer, r in [-0.5, 0.5]
    4. Compute 2^r using polynomial approximation
    5. Scale by 2^k: result = 2^k * 2^r

    IMPORTANT: All intermediate computations use ONLY fp32 precision.
    """

    ttnn_dtype = x.dtype
    x = ttnn.to_torch(x, dtype=torch.float32)
    x = x.numpy()

    x = np.float32(x)

    # Constants (all in fp32)
    INV_LN2 = np.float32(1.4426950408889634)  # 1/ln(2)
    OVERFLOW_THRESHOLD = np.float32(88.0)
    UNDERFLOW_THRESHOLD = np.float32(-103.97)
    HALF = np.float32(0.5)
    ONE = np.float32(1.0)

    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=np.float32)

    # Handle special cases (vectorized)
    nan_mask = np.isnan(x)
    overflow_mask = x >= OVERFLOW_THRESHOLD
    underflow_mask = x <= UNDERFLOW_THRESHOLD
    
    # Set special case values
    result[nan_mask] = np.float32(np.nan)
    result[overflow_mask] = np.float32(np.inf)
    result[underflow_mask] = np.float32(0.0)
    
    # Mask for normal computation (not special cases)
    normal_mask = ~(nan_mask | overflow_mask | underflow_mask)

    # Only compute for normal values
    if np.any(normal_mask):
        x_normal = x[normal_mask]

        # Step 1: Convert to base-2 exponential
        # z = x / ln(2)
        z = np.float32(x_normal * INV_LN2)

        # Step 2: Range reduction
        # Split z = k + r where k is integer, r in [-0.5, 0.5]
        k_temp = np.float32(z + HALF)
        k = np.float32(np.floor(k_temp))
        k_int = k.astype(np.int32)  # Convert to integer array

        # BUG FIX: Compute r = z - k (NOT r = x - k*ln2)
        # This is correct because we want: 2^z = 2^(k+r) = 2^k * 2^r
        r = np.float32(z - k)

        # Step 3: Polynomial approximation for 2^r
        # Using Taylor series: 2^r = exp(r*ln2) = 1 + r*ln2 + (r*ln2)²/2! + ...
        # For better accuracy, compute s = r*ln2, then use exp(s) polynomial

        LN2 = np.float32(0.6931471805599453)

        # Compute s = r * ln(2), all in fp32
        s = np.float32(r * LN2)

        # Now approximate exp(s) using Taylor series
        # exp(s) = 1 + s + s²/2! + s³/3! + s⁴/4! + s⁵/5! + s⁶/6!
        # Coefficients for exp(s)
        # 1 + x * (1 + x * (0.5 + x * (0.16666664183139801025390625 + x * (4.16664592921733856201171875e-2 + x * (8.333896286785602569580078125e-3 + x * (1.39354146085679531097412109375e-3 + x * 1.95693559362553060054779052734375e-4))))))
        
        c7 = np.float32(1.95693559362553060054779052734375e-4)     # 1/7!
        c6 = np.float32(1.39354146085679531097412109375e-3)     # 1/6!
        c5 = np.float32(8.333896286785602569580078125e-3)
        c4 = np.float32(4.16664592921733856201171875e-2)
        c3 = np.float32(0.16666664183139801025390625)
        c2 = np.float32(0.5)
        c1 = np.float32(1.0)
        c0 = np.float32(1.0)


        # Horner's method for better accuracy (vectorized)
        # p = c0 + s*(c1 + s*(c2 + s*(c3 + s*(c4 + s*(c5 + s*c6)))))
        p = np.float32(c7)

        temp = np.float32(p * s)
        p = np.float32(c6 + temp)

        temp = np.float32(p * s)
        p = np.float32(c5 + temp)

        temp = np.float32(p * s)
        p = np.float32(c4 + temp)

        temp = np.float32(p * s)
        p = np.float32(c3 + temp)

        temp = np.float32(p * s)
        p = np.float32(c2 + temp)

        temp = np.float32(p * s)
        p = np.float32(c1 + temp)

        temp = np.float32(p * s)
        p = np.float32(c0 + temp)

        



        # Step 4: Scale by 2^k using ldexp (fp32 only, vectorized)
        # ldexp is just bit manipulation of exponent, works fine on fp32
        result_normal = np.float32(np.ldexp(p, k_int))
        
        # Assign computed values back to result array
        result[normal_mask] = result_normal

    result = torch.from_numpy(result)
    result = ttnn.from_torch(result, dtype=ttnn_dtype)

    return result

def exp_claude2(x):
    """
    Compute exp(x) using Cody-Waite range reduction for improved accuracy.
    Vectorized to handle arrays element-wise.

    Algorithm:
    1. Handle special cases (overflow, underflow)
    2. Convert to base-2: exp(x) = 2^(x/ln2)
    3. Range reduction using Cody-Waite: compute k, then r = x - k*ln2_hi - k*ln2_lo
    4. Compute 2^r using polynomial approximation
    5. Scale by 2^k: result = 2^k * 2^r

    IMPORTANT: All intermediate computations use ONLY fp32 precision.
    """
    ttnn_dtype = x.dtype
    x = ttnn.to_torch(x, dtype=torch.float32)
    x = x.numpy()

    x = np.float32(x)

    # Constants (all in fp32)
    INV_LN2 = np.float32(1.4426950408889634)  # 1/ln(2)

    # Cody-Waite constants: ln(2) split into high and low parts
    # ln(2) ≈ LN2_HI + LN2_LO
    # LN2_HI has lower 12 bits zeroed for exact multiplication
    LN2_HI = np.float32(0.6931152343750000)   # High bits of ln(2)
    LN2_LO = np.float32(3.19461832987e-05)    # Low bits of ln(2)

    OVERFLOW_THRESHOLD = np.float32(88.0)
    UNDERFLOW_THRESHOLD = np.float32(-103.97)
    HALF = np.float32(0.5)
    ONE = np.float32(1.0)

    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=np.float32)

    # Handle special cases (vectorized)
    nan_mask = np.isnan(x)
    overflow_mask = x >= OVERFLOW_THRESHOLD
    underflow_mask = x <= UNDERFLOW_THRESHOLD
    
    # Set special case values
    result[nan_mask] = np.float32(np.nan)
    result[overflow_mask] = np.float32(np.inf)
    result[underflow_mask] = np.float32(0.0)
    
    # Mask for normal computation (not special cases)
    normal_mask = ~(nan_mask | overflow_mask | underflow_mask)

    # Only compute for normal values
    if np.any(normal_mask):
        x_normal = x[normal_mask]

        # Step 1: Compute k = round(x / ln(2))
        z = np.float32(x_normal * INV_LN2)
        # Use floor(z + 0.5) for rounding to nearest integer (works for all z)
        k_temp = np.float32(z + HALF)
        k = np.float32(np.floor(k_temp))
        k_int = k.astype(np.int32)  # Convert to integer array

        # Step 2: Cody-Waite range reduction
        # Compute r = x - k*ln(2) in extended precision
        # r = x - k*LN2_HI - k*LN2_LO

        # First subtract k * LN2_HI
        temp1 = np.float32(k * LN2_HI)
        r_hi = np.float32(x_normal - temp1)

        # Then subtract k * LN2_LO
        temp2 = np.float32(k * LN2_LO)
        r = np.float32(r_hi - temp2)

        # Step 3: Polynomial approximation for 2^r where r ≈ r / ln(2)
        # We need to compute 2^r = exp(r * ln(2))
        # Let s = r (since r is already the reduced value)
        # We want exp(r * ln(2)), so compute polynomial on s * ln(2)

        # Actually, we have r = x - k*ln(2), so:
        # exp(x) = exp(k*ln(2) + r) = 2^k * exp(r)

        # Compute exp(r) using Taylor series
        # exp(r) = 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5! + r⁶/6! + r⁷/7!

        # Use 7th order polynomial for better accuracy
        c7 = np.float32(1.0 / 5040.0)    # 1/7!
        c6 = np.float32(1.0 / 720.0)     # 1/6!
        c5 = np.float32(1.0 / 120.0)     # 1/5!
        c4 = np.float32(1.0 / 24.0)      # 1/4!
        c3 = np.float32(1.0 / 6.0)       # 1/3!
        c2 = np.float32(0.5)             # 1/2!
        c1 = np.float32(1.0)             # 1/1!
        c0 = np.float32(1.0)             # constant

        # Horner's method (vectorized)
        p = np.float32(c7)

        temp = np.float32(p * r)
        p = np.float32(c6 + temp)

        temp = np.float32(p * r)
        p = np.float32(c5 + temp)

        temp = np.float32(p * r)
        p = np.float32(c4 + temp)

        temp = np.float32(p * r)
        p = np.float32(c3 + temp)

        temp = np.float32(p * r)
        p = np.float32(c2 + temp)

        temp = np.float32(p * r)
        p = np.float32(c1 + temp)

        temp = np.float32(p * r)
        p = np.float32(c0 + temp)

        # Step 4: Scale by 2^k using ldexp (fp32 only, vectorized)
        # ldexp is just bit manipulation of exponent, works fine on fp32
        result_normal = np.float32(np.ldexp(p, k_int))
        
        # Assign computed values back to result array
        result[normal_mask] = result_normal

    result = torch.from_numpy(result)
    result = ttnn.from_torch(result, dtype=ttnn_dtype)

    return result


def float_to_bits(f):
    """Convert float32 to its bit representation as uint32."""
    return struct.unpack('>I', struct.pack('>f', f))[0]


def bits_to_float(b):
    """Convert uint32 bit representation to float32."""
    return struct.unpack('>f', struct.pack('>I', b))[0]

def ln_fp32(x):
    """
    Compute natural logarithm using only fp32 internal arithmetic.
    Vectorized to handle arrays element-wise.

    Algorithm:
    1. Handle special cases (x <= 0, infinity, NaN)
    2. Extract exponent and mantissa: x = 2^n × m
    3. Reduce range: adjust m to be in [sqrt(2)/2, sqrt(2)]
    4. Compute ln(m) using polynomial approximation
    5. Return n×ln(2) + ln(m)

    IMPORTANT: All intermediate computations use ONLY fp32 precision.
    """

    ttnn_dtype = x.dtype
    x = ttnn.to_torch(x, dtype=torch.float32)
    x = x.numpy()

    x = np.float32(x)

    # Constants (precomputed in fp32)
    LN2 = np.float32(0.6931471805599453)  # ln(2)
    SQRT2 = np.float32(1.4142135623730951)  # sqrt(2)
    HALF = np.float32(0.5)
    ONE = np.float32(1.0)
    TWO = np.float32(2.0)
    ZERO = np.float32(0.0)
    BIAS = np.float32(127.0)

    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=np.float32)

    # Handle special cases (vectorized)
    nan_mask = np.isnan(x)
    neg_mask = x < ZERO
    zero_mask = x == ZERO
    inf_mask = np.isinf(x)
    
    # Set special case values
    result[nan_mask] = np.float32(np.nan)
    result[neg_mask] = np.float32(np.nan)
    result[zero_mask] = np.float32(-np.inf)
    result[inf_mask] = np.float32(np.inf)
    
    # Mask for normal computation (not special cases)
    normal_mask = ~(nan_mask | neg_mask | zero_mask | inf_mask)

    # Only compute for normal values
    if np.any(normal_mask):
        x_normal = x[normal_mask]

        # Extract bits using view casting (vectorized)
        # NOTE: This is acceptable as it's just bit manipulation
        # not higher precision arithmetic. We're extracting the IEEE 754 components.
        bits = x_normal.view(np.uint32)

        # Extract exponent (biased by 127) - convert to fp32 immediately (vectorized)
        exp_biased = np.float32((bits >> 23) & 0xFF)
        exp = np.float32(exp_biased - BIAS)

        # Extract mantissa and construct m in [1, 2) (vectorized)
        mantissa_bits = (bits & 0x007FFFFF) | 0x3F800000
        m = mantissa_bits.view(np.float32).astype(np.float32)

        # Range reduction: if m >= sqrt(2), divide by 2 and increment exponent
        # All operations in fp32 (vectorized)
        sqrt2_mask = m >= SQRT2
        m = np.where(sqrt2_mask, np.float32(m * HALF), m)
        exp = np.where(sqrt2_mask, np.float32(exp + ONE), exp)

        # Now m is in [sqrt(2)/2, sqrt(2)] ≈ [0.707, 1.414]
        # Transform to z = (m - 1) / (m + 1) for better convergence
        # This maps m ∈ [0.707, 1.414] to z ∈ [-0.172, 0.172]
        # ln(m) = 2 × (z + z³/3 + z⁵/5 + z⁷/7 + ...)

        # ALL operations happen on fp32 values (vectorized)
        m_minus_1 = np.float32(m - ONE)
        m_plus_1 = np.float32(m + ONE)
        z = np.float32(m_minus_1 / m_plus_1)
        z2 = np.float32(z * z)

        # Polynomial approximation using odd powers
        # ln(m) = 2z(1 + z²/3 + z⁴/5 + z⁶/7 + z⁸/9 + z¹⁰/11)
        # Using Horner's method with ALL operations in fp32 (vectorized)

        # Coefficients for the series (precomputed as fp32)
        c11 = np.float32(0.09090909090909091)  # 1/11
        c9 = np.float32(0.1111111111111111)    # 1/9
        c7 = np.float32(0.14285714285714285)   # 1/7
        c5 = np.float32(0.2)                    # 1/5
        c3 = np.float32(0.3333333333333333)    # 1/3

        # Horner's method - each operation happens in fp32 before next (vectorized)
        p = np.float32(c11)

        temp = np.float32(z2 * p)
        p = np.float32(c9 + temp)

        temp = np.float32(z2 * p)
        p = np.float32(c7 + temp)

        temp = np.float32(z2 * p)
        p = np.float32(c5 + temp)

        temp = np.float32(z2 * p)
        p = np.float32(c3 + temp)

        temp = np.float32(z2 * p)
        p = np.float32(ONE + temp)

        # Final computation: ln(m) = 2 * z * p
        temp = np.float32(z * p)
        ln_m = np.float32(TWO * temp)

        # Combine: ln(x) = exp×ln(2) + ln(m)
        temp = np.float32(exp * LN2)
        result_normal = np.float32(temp + ln_m)
        
        # Assign computed values back to result array
        result[normal_mask] = result_normal

    result = torch.from_numpy(result)
    result = ttnn.from_torch(result, dtype=ttnn_dtype)

    return result


# No golden function set => use ttnn.get_golden_function() from first implementation
UNARY_OPERATIONS = {
    "abs": {
        "implementations": {
            "abs": ttnn.abs
        },
        "golden": torch.abs
    },
    "identity": {
        "implementations": {
            "identity": ttnn.identity 
        },
    },
    "fill": {
        "implementations": {
            "fill": lambda x, output_tensor: ttnn.fill(x, 1.99999988079071044921875, output_tensor=output_tensor)
        },
        "golden": lambda x, out: torch.fill(out, 1.99999988079071044921875)
    },
    "exp": {
        "implementations": {
            "exp": ttnn.exp,
            "exp-approx": lambda x, output_tensor: generic_unary_kernel(generate_kernel_source_code_from_llk("unary", "exp_tile_init<true, false>", "exp_tile<true, false>"), x, output_tensor),
            "exp-fast-approx": lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
            },
        "golden": torch.exp,
    },
    "exp2": {
        "implementations": {
            "exp2": ttnn.exp2
        },
    },
    "expm1": {
        "implementations": {
            "expm1": ttnn.expm1,
            # "expm1-new": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("expm1-v1"), x, output_tensor),
        },
    },
    "tanh": {
        "implementations": {
            "tanh": ttnn.tanh,
            "tanh-approx": lambda x, output_tensor: ttnn.tanh(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
            "tanh-cf": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("tanh-v1"), x, output_tensor),
            "tanh-pade-5,5":  lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("tanh-pade-5,5"), x, output_tensor),
            "tanh-minimax-v1[6]": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("tanh-minimax-v1[6]"), x, output_tensor),
            },
        "golden": torch.tanh
    },
    "hardtanh": {
        "implementations": {
            "hardtanh": lambda x, output_tensor: ttnn.hardtanh(x)
        },
        "golden": lambda x, out: torch.nn.functional.hardtanh(x)
    },
    "cosh": {
        "implementations": {
            "cosh": lambda x, output_tensor: ttnn.cosh(x)
        },
        "golden": torch.cosh
    },
    "sinh": {
        "implementations": {
            "sinh": lambda x, output_tensor: ttnn.sinh(x)
        },
        "golden": torch.sinh
    },
    "log": {
        "implementations": {
            "log": ttnn.log,
        },
    },
    "log10": {
        "implementations": {
            "log10": ttnn.log10
        },
    },
    "log2": {
        "implementations": {
            "log2": ttnn.log2,
        },
    },
    "log1p": {
        "implementations": {
            "log1p": ttnn.log1p
        },
    },
    "relu": {
        "implementations": {
            "relu": ttnn.relu
        },
    },
    "relu_max": {
        "implementations": {
            "relu_max": lambda x, output_tensor: ttnn.relu_max(x, output_tensor=output_tensor, upper_limit=1.0)
        },
        "golden": lambda x, out: torch.minimum(torch.nn.functional.relu(x), torch.full_like(x, 1.0)) 
    },
    "relu_min": {
        "implementations": {
            "relu_min": lambda x, output_tensor: ttnn.relu_min(x, output_tensor=output_tensor, lower_limit=1.0)
        },
        "golden": lambda x, out: torch.maximum(torch.nn.functional.relu(x), torch.full_like(x, 1.0)) 
    },
    "silu": {
        "implementations": {
            "silu": ttnn.silu,
            "swish": lambda x, output_tensor: ttnn.swish(x)  # swish is same as silu
        },
    },
    "gelu": {
        "implementations": {
            "gelu": ttnn.gelu,
            "gelu_approx": lambda x, output_tensor: ttnn.gelu(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
            "gelu-tanh": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("gelu-tanh"), x, output_tensor),
        },
    },
    "logit": {
        "implementations": {
            "logit": lambda x, output_tensor: ttnn.logit(x)
        },
        "golden": torch.logit
    },
    "mish": {
        "implementations": {
            "mish": ttnn.mish
        },
    },
    "hardmish": {
        "implementations": {
            "hardmish": ttnn.hardmish,
            "hardmish-fast": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("hardmish"), x, output_tensor)
        },
    },
    "elu": {
        "implementations": {
            "elu": lambda x, output_tensor: ttnn.elu(x, output_tensor=output_tensor, alpha=1.0)
        },
        "golden": lambda x, out: torch.nn.functional.elu(x)
    },
    "celu": {
        "implementations": {
            "celu": lambda x, output_tensor: ttnn.celu(x, output_tensor=output_tensor, alpha=1.0)
        },
        "golden": lambda x, out: torch.nn.functional.celu(x)
    },
    "sigmoid": {
        "implementations": {
            "sigmoid": ttnn.sigmoid,
            "sigmoid-approx": lambda x, output_tensor: ttnn.sigmoid(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
            "sigmoid-accurate": ttnn.sigmoid_accurate,
            "sigmoid-accurate-approx": lambda x, output_tensor: ttnn.sigmoid_accurate(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
            "sigmoid-21f": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("sigmoid"), x, output_tensor),
        },
    },
    "log_sigmoid": {
        "implementations": {
            "log_sigmoid": ttnn.log_sigmoid
        },
    },
    "selu": {
        "implementations": {
            "selu": lambda x, output_tensor: ttnn.selu(x)
        },
        "golden": lambda x, out: torch.nn.functional.selu(x)
    },
    "softplus": {
        "implementations": {
            "softplus": ttnn.softplus,
            "softplus-log1pexp": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("softplus-log1pexp"), x, output_tensor),
            "softplus-minimax-v1[5]": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_polynomial("softplus-poly", [-1.88397825695574283599853515625e-4, -1.717669540084898471832275390625e-3, 5.2007441408932209014892578125e-3, 0.113285191357135772705078125, 0.476007401943206787109375, 0.689675033092498779296875]), x, output_tensor),
            "softplus-minimax-v1[8]": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_polynomial("softplus-poly", [7.496111464888599584810435771942138671875e-8, 7.1853546614875085651874542236328125e-6, 5.31854093424044549465179443359375e-5, -2.879406674765050411224365234375e-4, -3.30807245336472988128662109375e-3, 3.11028095893561840057373046875e-3, 0.121423818171024322509765625, 0.4927570819854736328125, 0.692435443401336669921875]), x, output_tensor),
        },
    },
    "softsign": {
        "implementations": {
            "softsign": lambda x, output_tensor: ttnn.softsign(x)
        },
        "golden": lambda x, out: torch.nn.functional.softsign(x)
    },
    "tan": {
        "implementations": {
            "tan": ttnn.tan
        },
    },
    "atan": {
        "implementations": {
            "atan": ttnn.atan
        },
    },
    "sin": {
        "implementations": {
            "sin": ttnn.sin
        },
    },
    "cos": {
        "implementations": {
            "cos": ttnn.cos
        },
    },
    "sqrt": {
        "implementations": {
            "sqrt": ttnn.sqrt
        },
    },
    "cbrt": {
        "implementations": {
            "cbrt": lambda x, output_tensor: ttnn.cbrt(x),
        },
        "golden": lambda x, out: torch.pow(x, 1/3)
    },
    "rsqrt": {
        "implementations": {
            "rsqrt": ttnn.rsqrt
        },
    },
    "reciprocal": {
        "implementations": {
            "reciprocal": ttnn.reciprocal
        },
    },
    "digamma": {
        "implementations": {
            "digamma": lambda x, output_tensor: ttnn.digamma(x)
        },
        "golden": torch.digamma
    },
    "lgamma": {
        "implementations": {
            "lgamma": lambda x, output_tensor: ttnn.lgamma(x)
        },
        "golden": torch.lgamma
    },
    "tanhshrink": {
        "implementations": {
            "tanhshrink": lambda x, output_tensor: ttnn.tanhshrink(x)
        },
        "golden": lambda x, out: torch.nn.functional.tanhshrink(x)
    },
    "erfinv": {
        "implementations": {
            "erfinv": lambda x, output_tensor: ttnn.erfinv(x)
        },
        "golden": lambda x, out: torch.special.erfinv(x)
    },
}


def divide_sfpu(x, y):
    assert isinstance(x, ttnn.Tensor)
    assert isinstance(y, ttnn.Tensor)

    # Convert on host to ensure proper rounding
    # device = x.device
    layout = x.layout

    global global_device
    device = global_device
    assert device is not None

    host_bf16_x = ttnn.to_torch(x)
    host_bf16_y = ttnn.to_torch(y)

    # Convert to float32
    host_f32_x = host_bf16_x.to(torch.float32)
    host_f32_y = host_bf16_y.to(torch.float32)

    ttnn_x = ttnn.from_torch(host_f32_x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_y = ttnn.from_torch(host_f32_y, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.divide(ttnn_x, ttnn_y)

    # Convert back to bf16
    result = ttnn.to_torch(result)
    result = result.to(torch.bfloat16)
    result = ttnn.from_torch(result, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return result


BINARY_OPERATIONS = {
    "add": {
        "implementations": {
            "add": ttnn.add
        },
        "golden": torch.add
    },
    "multiply": {
        "implementations": {
            "multiply": ttnn.multiply
        },
        "golden": torch.multiply
    },
    "hypot": {
        "implementations": {
            "hypot": ttnn.hypot
        },
        "golden": torch.hypot
    },
    "pow": {
        "implementations": {
            "pow": ttnn.pow
        },
        "golden": torch.pow
    },
    "divide": {
        "implementations": {
            "divide": ttnn.divide,
            "div": ttnn.div,
            "div-accurate": lambda x, y: ttnn.div(x, y, accurate_mode=True)
        },
        "golden": torch.div
    },
    "fmod": {
        "implementations": {
            "fmod": ttnn.fmod
        },
        "golden": torch.fmod
    },
    "atan2": {
        "implementations": {
            "atan2": ttnn.atan2
        },
        "golden": torch.atan2
    },
    "rsub": {
        "implementations": {
            "rsub": ttnn.rsub
        },
        "golden": torch.rsub
    }
}


def get_op_implementations(operation_dict, category):

    if not category in operation_dict:
        raise ValueError(f"Category {category} not found in operation_dict")

    return operation_dict[category]

def get_golden_function(operation_dict, operation_name: str):
    golden_function = None
    operation = get_op_implementations(operation_dict, operation_name)
    if "golden" in operation:
        golden_function = operation["golden"]
    else:
        for impl_name, implementation in operation["implementations"].items():
            try:
                golden_function = ttnn.get_golden_function(implementation)
            except:
                print(f"{TERM_RED}No golden function found for implementation {impl_name}{TERM_RESET}")
                continue
            
            golden_function = implementation.golden_function
            break

    if golden_function is None:
        raise ValueError(f"No golden operation found for operation {operation_name}")

    return golden_function
    

def iterate_all_operations(operation_dict):
    """ Iterate over all operations in the operation dictionary 
    and yield the implementation name, base operation name the implementation function, and the golden function
    """
    for base_operation_name, operations in operation_dict.items():

        golden_function = get_golden_function(operation_dict, base_operation_name)
        for impl_name, implementation in operations["implementations"].items():
            yield (impl_name, base_operation_name, implementation, golden_function)

def get_operation_variant_by_name(operation_dict, impl_name):

    for category, operations in operation_dict.items():
        if impl_name in operations["implementations"].keys():

            impl = operations["implementations"][impl_name]

            if "golden" in operations:
                golden_operation = operations["golden"]
            else:
                ttnn_impl = getattr(ttnn, category)
                golden_operation = ttnn.get_golden_function(ttnn_impl)

            return impl, golden_operation

    return None



def run_ttnn_op(fun, args, device):
    global global_device
    global_device = device

    ttnn_args = [convert_to_ttn(arg, device) for arg in args if isinstance(arg, torch.Tensor)]

    return fun(*ttnn_args)