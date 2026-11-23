from comfy_kitchen.tensor import QuantizedLayout, TensorCoreFP8Layout
from comfy_kitchen.tensor import QuantizedTensor as QuantizedTensor
from comfy_kitchen.tensor import TensorWiseINT8Layout as TensorWiseINT8Layout

class TensorCoreFP8E5M2Layout(TensorCoreFP8Layout): ...

QUANT_ALGOS: dict[str, dict[str, object]]

def register_layout_op(torch_op: object, layout_cls: type[QuantizedLayout]) -> object: ...
