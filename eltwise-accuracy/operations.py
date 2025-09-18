import ttnn
import torch
import math


def run_op_fp32(ttnn_op, args, device=None):
    ttnn_args = [convert_to_ttn(arg, device) for arg in args]
    return ttnn_op(*ttnn_args)


UNARY_OPERATIONS = {
    # Exponential functions
    "exp": (torch.exp, ttnn.exp, math.exp, "exp"),
    "exp-approx": (
        torch.exp,
        lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
        "exp",
    ),
    "exp-fast-approx": (
        torch.exp,
        lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
        "exp",
    ),
    "exp-fast-approx-v2": (
        torch.exp,
        lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
        "exp",
    ),
    "exp_cond": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    "exp_approx0": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    "exp21f": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    "exp_21f_round_nearest": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    "exp_hybrid": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    # "exp_accurate_python": (
    #     torch.exp,
    #     exp_accurate_python,
    #     None,
    #     "exp",
    # ),
    # "exp_python_alt1": (
    #     torch.exp,
    #     lambda x, output_tensor: exp_accurate_python(x, output_tensor, exp_regression=exp_regression_0p5_to_1_alt1),
    #     None,
    #     "exp",
    # ),
    "tanh": (
        torch.tanh,
        ttnn.tanh,
        math.tanh,
        "tanh",
    ),  # ttnn.tanh() does not support output_tensor ?
    "tanh_accurate": (
        torch.tanh,
        lambda x, output_tensor: ttnn.tanh(x, accuracy=True, output_tensor=output_tensor),
        math.tanh,
        "tanh",
    ),
    "cosh": (
        torch.cosh,
        lambda x, output_tensor: ttnn.cosh(x),
        math.cosh,
        "cosh",
    ),  # ttnn.cosh() does not support output_tensor ?
    "sinh": (
        torch.sinh,
        lambda x, output_tensor: ttnn.sinh(x),
        math.sinh,
        "sinh",
    ),  # ttnn.sinh() does not support output_tensor ?
    # Logarithmic functions
    "log": (torch.log, ttnn.log, math.log, "log"),
    "log10": (torch.log10, ttnn.log10, math.log10, "log10"),
    "log2": (torch.log2, ttnn.log2, math.log2, "log2"),
    "log1p": (torch.log1p, ttnn.log1p, math.log1p, "log1p"),
    "logaddexp": (torch.logaddexp, ttnn.logaddexp, None, "logaddexp"),
    "logaddexp2": (torch.logaddexp2, ttnn.logaddexp2, None, "logaddexp2"),
    # Activation functions
    "silu": (lambda x, out: torch.nn.SiLU()(x), ttnn.silu, None, "silu"),
    "gelu": (lambda x, out: torch.nn.GELU()(x), ttnn.gelu, None, "gelu"),
    "gelu_approx": (
        lambda x, out: torch.nn.GELU()(x),
        lambda x, output_tensor: ttnn.gelu(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
        "gelu",
    ),
    "logit": (
        torch.logit,
        lambda x, output_tensor: ttnn.logit(x),
        None,
        "logit",
    ),  # ttnn.logit does not support output_tensor ?
    "swish": (
        lambda x, out: torch.nn.SiLU()(x),
        lambda x, output_tensor: ttnn.swish(x),
        None,
        "swish",
    ),  # ttnn.swish does not support output_tensor ?
    "mish": (lambda x, out: torch.nn.Mish()(x), ttnn.mish, None, "mish"),
    "elu": (
        lambda x, out: torch.nn.ELU()(x),
        lambda x, output_tensor: ttnn.elu(x, output_tensor=output_tensor, alpha=1.0),
        None,
        "elu",
    ),  # Unlike torch, ttnn.elu does not use alpha=1 by default
    "selu": (
        lambda x, out: torch.nn.SELU()(x),
        lambda x, output_tensor: ttnn.selu(x),
        None,
        "selu",
    ),  # ttnn.selu does not support output_tensor ?
    "softplus": (lambda x, out: torch.nn.Softplus()(x), ttnn.softplus, None, "softplus"),
    "softsign": (
        lambda x, out: torch.nn.Softsign()(x),
        lambda x, output_tensor: ttnn.softsign(x),
        None,
        "softsign",
    ),  # ttnn.softsign does not support output_tensor ?
    # Trigonometric functions
    "tan": (torch.tan, ttnn.tan, math.tan, "tan"),
    "atan": (torch.atan, ttnn.atan, math.atan, "atan"),
    "atan2": (torch.atan2, ttnn.atan2, math.atan2, "atan2"),
    "sin": (torch.sin, ttnn.sin, math.sin, "sin"),
    "cos": (torch.cos, ttnn.cos, math.cos, "cos"),
    # Miscellaneous functions
    "sqrt": (torch.sqrt, ttnn.sqrt, math.sqrt, "sqrt"),
    "rsqrt": (
        torch.rsqrt,
        lambda x, output_tensor: ttnn.rsqrt(x, fast_and_approximate_mode=False, output_tensor=output_tensor),
        None,
        "rsqrt",
    ),
    "rsqrt_approx": (
        torch.rsqrt,
        lambda x, output_tensor: ttnn.rsqrt(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
        "rsqrt",
    ),
    "reciprocal": (
        torch.reciprocal,
        ttnn.reciprocal,
        None,
        "reciprocal",
    ),
    "digamma": (
        torch.digamma,
        lambda x, output_tensor: ttnn.digamma(x),
        None,
        "digamma",
    ),  # ttnn.digamma does not support output_tensor ?
    "lgamma": (
        torch.lgamma,
        lambda x, output_tensor: ttnn.lgamma(x),
        math.lgamma,
        "lgamma",
    ),  # ttnn.lgamma does not support output_tensor ?
    "tanhshrink": (
        lambda x, out: torch.nn.Tanhshrink()(x),
        lambda x, output_tensor: ttnn.tanhshrink(x),
        None,
        "tanhshrink",
    ),  # ttnn.tan
}


global_device = None

def divide_sfpu(x, y):
    assert isinstance(x, ttnn.Tensor)
    assert isinstance(y, ttnn.Tensor)

    # Convert on host to ensure proper rounding
    # device = x.device
    layout = x.layout

    global global_device
    device = global_device
    assert device is not None

    print(f"device =\n{device}")
    print(f"layout =\n{layout}")

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
    "pow": {
        "ttnn": ttnn.pow,
        "torch": torch.pow,
    },
    "pow21f": {
        "ttnn": ttnn.pow,
        "torch": torch.pow,
    },
    "divide": {
        "ttnn": ttnn.divide,
        "torch": torch.div,
    },
    "divide-sfpu": {
        "ttnn": lambda x, y: divide_sfpu(x, y),
        "torch": torch.div,
    },
    "div": {
        "ttnn": ttnn.div,
        "torch": torch.div,
    },
    "div-accurate": {
        "ttnn": lambda x, y: ttnn.div(x, y, accurate_mode=True),
        "torch": torch.div,
    },
}


def convert_to_ttn(tensor, device):

    if isinstance(tensor, ttnn.Tensor):
        return tensor

    # Shard data on all cores to speed-up computation
    ttnn_tensor = ttnn.from_torch(tensor, layout=ttnn.Layout.TILE, device=device)
    
    return ttnn_tensor


def run_ttnn_op(fun, args, device):
    global global_device
    global_device = device

    ttnn_args = [convert_to_ttn(arg, device) for arg in args if isinstance(arg, torch.Tensor)]

    return fun(*ttnn_args)
