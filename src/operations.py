import ttnn
import torch
from .utils import TERM_RED, TERM_RESET

import mpmath as mp


from .kernel_generator import generate_unary_kernel_from_polynomial, generate_unary_kernel_from_sfpi_source, generic_unary_kernel, generate_kernel_source_code_from_llk
from .kernel_generator import make_generic_binary_kernel_op, generic_binary_kernel_with_dst_init, generate_kernel_from_source_path


global_device = None


def gelu_mp(tensor, output_tensor):

    torch_tensor = tensor
    if isinstance(tensor, ttnn.Tensor):
        torch_tensor = ttnn.to_torch(tensor)

    ctx = mp.MPContext()
    ctx.prec = 300

    # Compute GELU(x)
    def local_gelu_lambda(x):
        mpx = ctx.mpf(x)
        res = ctx.mpf(0.5) * mpx * (ctx.erfc(-mpx / ctx.sqrt(ctx.mpf(2))))
        return float(res)

    torch_output_tensor = torch.clone(torch_tensor)
    torch_output_tensor.apply_(local_gelu_lambda)

    if isinstance(tensor, ttnn.Tensor):
        output_tensor = ttnn.from_torch(torch_output_tensor, layout=tensor.layout, device=tensor.device())
    else:
        output_tensor = torch_output_tensor

    return output_tensor

def gelu_torch(tensor, output_tensor):

    torch_tensor = tensor
    if isinstance(tensor, ttnn.Tensor):
        torch_tensor = ttnn.to_torch(tensor)

    input_dtype = torch_tensor.dtype
    torch_tensor = torch_tensor.to(dtype=torch.float64)

    torch_output = torch.nn.functional.gelu(torch_tensor)
    torch_output = torch_output.to(dtype=input_dtype)

    if isinstance(tensor, ttnn.Tensor):
        output_tensor = ttnn.from_torch(torch_output, layout=tensor.layout, device=tensor.device())
    else:
        output_tensor = torch_output

    return output_tensor


def run_op_fp32(ttnn_op, args, device=None):
    ttnn_args = [convert_to_ttn(arg, device) for arg in args]
    return ttnn_op(*ttnn_args)


def convert_to_ttn(tensor, device):

    if isinstance(tensor, ttnn.Tensor):
        return tensor

    # Shard data on all cores to speed-up computation
    ttnn_tensor = ttnn.from_torch(tensor, layout=ttnn.Layout.TILE, device=device)
    
    return ttnn_tensor


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
            # "exp-approx": lambda x, output_tensor: generic_unary_kernel(generate_kernel_source_code_from_llk("unary", "exp_tile_init<true, false>", "exp_tile<true, false>"), x, output_tensor),
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
        },
    },
    "tanh": {
        "implementations": {
            "tanh": ttnn.tanh,
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
        "golden": torch.log,
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
        },
    },
    "gelu": {
        "implementations": {
            "gelu": ttnn.gelu,
            "gelu_approx": lambda x, output_tensor: ttnn.gelu(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
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
    "acos": {
        "implementations": {
            "acos": ttnn.acos
        },
        "golden": torch.acos
    },
    "asin": {
        "implementations": {
            "asin": ttnn.asin
        },
        "golden": torch.asin
    },
    "atan": {
        "implementations": {
            "atan": ttnn.atan,
        },
        "golden": torch.atan
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
    "erf": {
        "implementations": {
            "erf": lambda x, output_tensor: ttnn.erf(x)
        },
        "golden": lambda x, out: torch.special.erf(x)
    },
    "erfinv": {
        "implementations": {
            "erfinv": lambda x, output_tensor: ttnn.erfinv(x)
        },
        "golden": lambda x, out: torch.special.erfinv(x)
    },
    "round": {
        "implementations": {
            "round": lambda x, output_tensor: ttnn.round(x, output_tensor=output_tensor)
        },
        "golden": torch.round
    },
    "ceil": {
        "implementations": {
            "ceil": lambda x, output_tensor: ttnn.ceil(x, output_tensor=output_tensor)
        },
        "golden": torch.ceil
    },
    "floor": {
        "implementations": {
            "floor": lambda x, output_tensor: ttnn.floor(x, output_tensor=output_tensor)
        },
        "golden": torch.floor
    },
}

BINARY_OPERATIONS = {
    "add": {
        "implementations": {
            "add": ttnn.add
        },
        "golden": torch.add
    },
    "subtract": {
        "implementations": {
            "subtract": ttnn.subtract
        },
        "golden": torch.subtract
    },
    "multiply": {
        "implementations": {
            "multiply": ttnn.multiply,
            "multiply-fp32acc": make_generic_binary_kernel_op("mul_tiles_init", "mul_tiles"),
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
            "atan2": ttnn.atan2,
        },
        "golden": torch.atan2
    },
    "rsub": {
        "implementations": {
            "rsub": ttnn.rsub
        },
        "golden": torch.rsub
    },
    "subtract": {
        "implementations": {
            "subtract": ttnn.subtract
        },
        "golden": torch.subtract
    },
    "multiply_accumulate": {
        "implementations": {
            "multiply_accumulate": lambda x, y: generic_binary_kernel_with_dst_init(generate_kernel_from_source_path("kernels/mul_with_dst_init.cpp"),x, y,-1.00000011920928955078125)
        },
        "golden": lambda x, y: torch.add(torch.mul(x, y), -1.00000011920928955078125)
    }
}


def make_unary_bw_golden(ttnn_bw_op):
    """Wrap a ttnn backward golden (grad, input) -> [tensor] into a unary golden (x, out=None) -> tensor.
    Uses grad=1 (ones_like), matching the TTNN implementation convention."""
    bw_golden = ttnn.get_golden_function(ttnn_bw_op)
    def golden(x, out=None):
        # Backward goldens call .backward() internally; re-enable grad in case
        # the measurement pipeline wrapped us in torch.no_grad().
        with torch.enable_grad():
            grad = torch.ones_like(x)
            x_req = x.detach().requires_grad_(True)
            result = bw_golden(grad, x_req)[0]
        if out is not None:
            out.copy_(result)
            return out
        return result
    return golden


def _bw_golden_from_torch(torch_op):
    """Fallback for backward ops whose ttnn golden requires hardware-specific kwargs (e.g., device)."""
    def golden(x, out=None):
        with torch.enable_grad():
            grad = torch.ones_like(x)
            x_req = x.detach().requires_grad_(True)
            y = torch_op(x_req)
            y.backward(gradient=grad)
            result = x_req.grad
        if out is not None:
            out.copy_(result)
            return out
        return result
    return golden


def _bw_impl(ttnn_bw_op):
    """Standard TTNN backward implementation with grad=1."""
    return lambda x, output_tensor: ttnn_bw_op(ttnn.ones_like(x), x)[0]


UNARY_BW_OPERATIONS = {
    # --- Misc ---
    "abs_bw":   {"implementations": {"abs_bw":   _bw_impl(ttnn.abs_bw)},   "golden": make_unary_bw_golden(ttnn.abs_bw)},
    "floor_bw": {"implementations": {"floor_bw": _bw_impl(ttnn.floor_bw)}, "golden": make_unary_bw_golden(ttnn.floor_bw)},

    # --- Exponentials ---
    "exp_bw":   {"implementations": {"exp_bw":   _bw_impl(ttnn.exp_bw)},   "golden": make_unary_bw_golden(ttnn.exp_bw)},
    "exp2_bw":  {"implementations": {"exp2_bw":  _bw_impl(ttnn.exp2_bw)},  "golden": make_unary_bw_golden(ttnn.exp2_bw)},
    "expm1_bw": {"implementations": {"expm1_bw": _bw_impl(ttnn.expm1_bw)}, "golden": make_unary_bw_golden(ttnn.expm1_bw)},

    # --- Logarithms ---
    "log_bw":   {"implementations": {"log_bw":   _bw_impl(ttnn.log_bw)},   "golden": make_unary_bw_golden(ttnn.log_bw)},
    "log10_bw": {"implementations": {"log10_bw": _bw_impl(ttnn.log10_bw)}, "golden": make_unary_bw_golden(ttnn.log10_bw)},
    "log2_bw":  {"implementations": {"log2_bw":  _bw_impl(ttnn.log2_bw)},  "golden": make_unary_bw_golden(ttnn.log2_bw)},
    "log1p_bw": {"implementations": {"log1p_bw": _bw_impl(ttnn.log1p_bw)}, "golden": make_unary_bw_golden(ttnn.log1p_bw)},

    # --- Square root ---
    "sqrt_bw":  {"implementations": {"sqrt_bw":  _bw_impl(ttnn.sqrt_bw)},  "golden": make_unary_bw_golden(ttnn.sqrt_bw)},
    "rsqrt_bw": {"implementations": {"rsqrt_bw": _bw_impl(ttnn.rsqrt_bw)}, "golden": make_unary_bw_golden(ttnn.rsqrt_bw)},

    # --- Trigonometric ---
    "sin_bw":  {"implementations": {"sin_bw":  _bw_impl(ttnn.sin_bw)},  "golden": make_unary_bw_golden(ttnn.sin_bw)},
    "cos_bw":  {"implementations": {"cos_bw":  _bw_impl(ttnn.cos_bw)},  "golden": make_unary_bw_golden(ttnn.cos_bw)},
    "tan_bw":  {"implementations": {"tan_bw":  _bw_impl(ttnn.tan_bw)},  "golden": make_unary_bw_golden(ttnn.tan_bw)},
    "asin_bw": {"implementations": {"asin_bw": _bw_impl(ttnn.asin_bw)}, "golden": make_unary_bw_golden(ttnn.asin_bw)},
    "acos_bw": {"implementations": {"acos_bw": _bw_impl(ttnn.acos_bw)}, "golden": make_unary_bw_golden(ttnn.acos_bw)},
    "atan_bw": {"implementations": {"atan_bw": _bw_impl(ttnn.atan_bw)}, "golden": make_unary_bw_golden(ttnn.atan_bw)},

    # --- Hyperbolic ---
    "sinh_bw":  {"implementations": {"sinh_bw":  _bw_impl(ttnn.sinh_bw)},  "golden": make_unary_bw_golden(ttnn.sinh_bw)},
    "cosh_bw":  {"implementations": {"cosh_bw":  _bw_impl(ttnn.cosh_bw)},  "golden": make_unary_bw_golden(ttnn.cosh_bw)},
    "tanh_bw":  {"implementations": {"tanh_bw":  _bw_impl(ttnn.tanh_bw)},  "golden": make_unary_bw_golden(ttnn.tanh_bw)},
    "asinh_bw": {"implementations": {"asinh_bw": _bw_impl(ttnn.asinh_bw)}, "golden": make_unary_bw_golden(ttnn.asinh_bw)},
    # acosh_bw golden requires a ttnn device kwarg for hardware nan handling; use torch directly.
    "acosh_bw": {"implementations": {"acosh_bw": _bw_impl(ttnn.acosh_bw)}, "golden": _bw_golden_from_torch(torch.acosh)},
    "atanh_bw": {"implementations": {"atanh_bw": _bw_impl(ttnn.atanh_bw)}, "golden": make_unary_bw_golden(ttnn.atanh_bw)},

    # --- Miscellaneous numeric ---
    "tanhshrink_bw": {"implementations": {"tanhshrink_bw": _bw_impl(ttnn.tanhshrink_bw)}, "golden": make_unary_bw_golden(ttnn.tanhshrink_bw)},
    "hardtanh_bw": {"implementations": {"hardtanh_bw": _bw_impl(ttnn.hardtanh_bw)}, "golden": make_unary_bw_golden(ttnn.hardtanh_bw)},
    "digamma_bw": {"implementations": {"digamma_bw": _bw_impl(ttnn.digamma_bw)}, "golden": make_unary_bw_golden(ttnn.digamma_bw)},
    "lgamma_bw": {"implementations": {"lgamma_bw": _bw_impl(ttnn.lgamma_bw)}, "golden": make_unary_bw_golden(ttnn.lgamma_bw)},
    "erfinv_bw": {"implementations": {"erfinv_bw": _bw_impl(ttnn.erfinv_bw)}, "golden": make_unary_bw_golden(ttnn.erfinv_bw)},

    # --- Activation functions ---
    "sigmoid_bw": {"implementations": {"sigmoid_bw": _bw_impl(ttnn.sigmoid_bw)}, "golden": make_unary_bw_golden(ttnn.sigmoid_bw)},
    "silu_bw":    {"implementations": {"silu_bw":    _bw_impl(ttnn.silu_bw)},    "golden": make_unary_bw_golden(ttnn.silu_bw)},
    "gelu_bw":    {"implementations": {"gelu_bw":    _bw_impl(ttnn.gelu_bw)},    "golden": make_unary_bw_golden(ttnn.gelu_bw)},
    "celu_bw":    {"implementations": {"celu_bw":    _bw_impl(ttnn.celu_bw)},    "golden": make_unary_bw_golden(ttnn.celu_bw)},
    "elu_bw":     {"implementations": {"elu_bw":     _bw_impl(ttnn.elu_bw)},     "golden": make_unary_bw_golden(ttnn.elu_bw)},
    "selu_bw":    {"implementations": {"selu_bw":    _bw_impl(ttnn.selu_bw)},    "golden": make_unary_bw_golden(ttnn.selu_bw)},
    "softplus_bw": {"implementations": {"softplus_bw": _bw_impl(ttnn.softplus_bw)}, "golden": make_unary_bw_golden(ttnn.softplus_bw)},
    "softsign_bw": {"implementations": {"softsign_bw": _bw_impl(ttnn.softsign_bw)}, "golden": make_unary_bw_golden(ttnn.softsign_bw)},
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
