import math
from dataclasses import dataclass
from typing import Iterator

from tabulate import tabulate
from torch import nn

__all__ = ["get_layer_names", "get_rf_data", "print_rf"]


@dataclass
class LayerRFData:
    depth: int
    name: str
    type: str
    output_shape: tuple[int, int] | None
    origin: tuple[float, float] | None
    jump: int | None
    receptive_field: int | None

    def values(self) -> tuple[tuple[int, int], tuple[float, float], int, int]:
        """Returns the numerical values as a tuple."""
        return (self.output_shape, self.origin, self.jump, self.receptive_field)

    def replace_values(self, values: tuple[tuple[int, int], tuple[float, float], int, int]) -> "LayerRFData":
        """Returns a new instance of `LayerRFData` with the numerical values replaced."""
        self.output_shape = values[0]
        self.origin = values[1]
        self.jump = values[2]
        self.receptive_field = values[3]
        return self


def iter_module(module: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Recursively iterates over all the modules in the given module."""

    for layer_name, layer in module.named_modules():
        yield layer_name, layer


def get_layer_names(module: nn.Module) -> list[str]:
    return [layer_name for layer_name, _ in iter_module(module)]


def is_custom_module(module: nn.Module) -> bool:
    """Returns whether the module is a custom module or not."""

    return len(list(module.children())) > 0


def get_rf_data(module: nn.Module, input_shape: tuple[int, int], max_depth: int = 99999) -> list[LayerRFData]:
    nx, ny = input_shape
    ox = oy = 0.5
    j = r = 1

    data: list[LayerRFData] = []
    for layer_name_dotted, layer in iter_module(module):
        layer_type = layer.__class__.__name__
        depth = 0 if layer_name_dotted == "" else len(layer_name_dotted.split("."))

        if depth == 0:
            layer_name = ""
        elif depth == 1:
            layer_name = f"├─ {layer_name_dotted}"
        else:
            layer_name = "|    " * (depth - 1) + f"└─ {layer_name_dotted.split('.')[-1]}"

        if isinstance(layer, nn.Sequential) or is_custom_module(layer):
            data.append(LayerRFData(depth, layer_name, layer_type, None, None, None, None))
            continue

        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
            p = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
            s = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
            k = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size

            nx = ((nx + 2 * p - k) / s) + 1
            ny = ((ny + 2 * p - k) / s) + 1
            if isinstance(module, nn.Conv2d):
                assert int(nx) == nx and int(ny) == ny
                nx, ny = int(nx), int(ny)
            else:
                nx, ny = math.floor(nx), math.floor(ny)

            r = r + (k - 1) * j

            ox = ox + (((k - 1) / 2) - p) * j
            oy = oy + (((k - 1) / 2) - p) * j

            j = j * s

        data.append(LayerRFData(depth, layer_name, layer_type, (nx, ny), (ox, oy), j, r))

    # Create cleaned version of the list where we obey `max_depth`
    data_clean = []
    for i, x in enumerate(data):
        if x.depth == max_depth:
            new_values = x.values()
            j = i + 1
            while len(data) > j and data[j].depth > max_depth:
                new_values = data[j].values()
                j += 1
            x = x.replace_values(new_values)

        if x.depth <= max_depth:
            data_clean.append(x)

    return data_clean


def print_rf(module: nn.Module, input_shape: tuple[int, int], max_depth: int = 99999):
    """
    Prints a table displaying receptive field data for an input of the given shape, for all layers in the given module.

    Args:
        module: a torch module (that has neither multiple branches nor skip connections).
        input_shape: the shape of an input for which you want to compute the receptive field
        max_depth:
            an optional max depth to print, where "depth" refers to the number of nested layers in your network.
    """

    data = get_rf_data(module, input_shape=input_shape, max_depth=max_depth)
    data = [(x.name, x.type, x.output_shape, x.origin, x.jump, x.receptive_field) for x in data]
    print(tabulate(data, headers=["Layer", "Type", "Output Shape", "Origin", "Jump", "Receptive Field"]))
