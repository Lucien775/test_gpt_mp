###########################################
############## FORMAT #####################
###########################################

import mptorch.quant as qpt
from mptorch import FloatingPoint
from typing import Literal

# Floating Point format
fp32 = FloatingPoint(exp=8, man=23, subnormals=True)
fpe5m10 = FloatingPoint(exp=5, man=10, subnormals=True)
fpe4m3 = FloatingPoint(exp=4, man=3, subnormals=True)

# Quantization
quant_fp32 = lambda x: x
quant_fpe5m10 = lambda x: qpt.float_quantize(
    x, exp=5, man=10, rounding="nearest", subnormals=True, saturate=True
)
quant_fpe4m3 = lambda x: qpt.float_quantize(
    x, exp=4, man=3, rounding="nearest", subnormals=True, saturate=True
)

# Layer Format
def create_layer_format(dtype: Literal['fp32','fp16', 'fp8']):
    r""" 
    Create layer format config using QAffineFormats
    Args:
        dtype:
            format of the layer, fp32, fp16(e5m10) or fp8 (e4m3) 
    """
    if dtype == "fp32":
        return qpt.QAffineFormats(
            weight_scaled_format=quant_fp32,
            bias_quant=quant_fp32,
            input_quant=quant_fp32,
            output_quant=quant_fp32,
            grad_quant=quant_fp32,
            s=0
        )
    elif dtype =="fp16":
        return qpt.QAffineFormats(
            fwd_mac=(fpe5m10,),
            fwd_rnd="nearest",
            bwd_mac=(fpe5m10,),
            bwd_rnd="nearest",
            weight_quant=quant_fpe5m10,
            bias_quant=quant_fpe5m10,
            input_quant=quant_fpe5m10,
            output_quant=quant_fpe5m10,
            grad_quant=quant_fpe5m10,
            s=1
        )
    elif dtype == "fp8":
        return qpt.QAffineFormats(
            fwd_mac=(fpe4m3,),
            fwd_rnd="nearest",
            bwd_mac=(fpe4m3,),
            bwd_rnd="nearest",
            weight_quant=quant_fpe4m3,
            bias_quant=quant_fpe4m3,
            input_quant=quant_fpe4m3,
            output_quant=quant_fpe4m3,
            grad_quant=quant_fpe4m3,
            s=2
        )
    else:
        raise ValueError(f"Unknown format type: {dtype}")
    
# Softmax format
def create_softmax_format(dtype: Literal['fp32','fp16', 'fp8']):
    r"""
    Create softmax format config using QSoftmaxFormat
    Args:
        dtype:
            format of the layer, fp32, fp16(e5m10) or fp8 (e4m3)
    """
    if dtype == "fp32":
        return qpt.QSoftmaxFormats(
            input_quant=quant_fp32,
            output_quant=quant_fp32,
            grad_quant=quant_fp32
        )
    elif dtype =="fp16":
        return qpt.QSoftmaxFormats(
            fwd_off=fpe5m10,
            fwd_exp=fpe5m10,
            fwd_acc=fpe5m10,
            fwd_lse=fpe5m10,
            bwd_add=fpe5m10,
            bwd_mul=fpe5m10,
            input_quant=quant_fpe5m10,
            output_quant=quant_fpe5m10,
            grad_quant=quant_fpe5m10
        )
    elif dtype == "fp8":
        return qpt.QSoftmaxFormats(
            fwd_off=fpe4m3,
            fwd_exp=fpe4m3,
            fwd_acc=fpe4m3,
            fwd_lse=fpe4m3,
            bwd_add=fpe4m3,
            bwd_mul=fpe4m3,
            input_quant=quant_fpe4m3,
            output_quant=quant_fpe4m3,
            grad_quant=quant_fpe4m3
        )
    else:
        raise ValueError(f"Unknown format type: {dtype}")
    
# Layer Norm format
def create_LN_format(dtype: Literal['fp32','fp16', 'fp8']):
    r"""
    Create layer norm format config using QLayerNormFormats
    Args:
        dtype:
            format of the layer, fp32, fp16(e5m10) or fp8 (e4m3)
    """
    if dtype == "fp32":
        return qpt.QLayerNormFormats(
            input_quant=quant_fp32,
            output_quant=quant_fp32,
            grad_quant=quant_fp32,
            weight_quant=quant_fp32,
            bias_quant=quant_fp32
        )
    elif dtype =="fp16":
        return qpt.QLayerNormFormats(
            fwd_acc=fpe5m10,
            fwd_mul=fpe5m10,
            fwd_div=fpe5m10,
            fwd_sqrt=fpe5m10,
            bwd_acc=fpe5m10,
            bwd_mul=fpe5m10,
            bwd_div=fpe5m10,
            input_quant=quant_fpe5m10,
            output_quant=quant_fpe5m10,
            grad_quant=quant_fpe5m10,
            weight_quant=quant_fpe5m10,
            bias_quant=quant_fpe5m10
        )
    elif dtype == "fp8":
        return qpt.QLayerNormFormats(
            fwd_acc=fpe4m3,
            fwd_mul=fpe4m3,
            fwd_div=fpe4m3,
            fwd_sqrt=fpe4m3,
            bwd_acc=fpe4m3,
            bwd_mul=fpe4m3,
            bwd_div=fpe4m3,
            input_quant=quant_fpe4m3,
            output_quant=quant_fpe4m3,
            grad_quant=quant_fpe4m3,
            weight_quant=quant_fpe4m3,
            bias_quant=quant_fpe4m3
        )
    else:
        raise ValueError(f"Unknown format type: {dtype}")
    