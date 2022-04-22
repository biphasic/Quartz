import quartz
import copy
import torch
import torch.nn as nn
from collections import OrderedDict
import sinabs.layers as sl


def from_model(
    model: nn.Module,
    t_max: int,
    add_spiking_output: bool = False,
):
    model = copy.deepcopy(model)

    if add_spiking_output:
        if isinstance(list(model.children())[-1], (nn.Conv2d, nn.Linear)):
            model.add_module("output", nn.ReLU())
        else:
            print(
                "Spiking output can only be added to sequential models that do not end in a ReLU. No layer has been added."
            )

    snn = OrderedDict()
    i = 0
    for name, module in list(model.named_children()):
        # if it's one of the layers we're looking for, substitute it
        if isinstance(module, nn.ReLU):
            snn.update(
                [
                    (
                        str(i),
                        quartz.IF(
                            t_max=t_max,
                            rectification=True,
                        ),
                    )
                ]
            )
            i += 1

        elif isinstance(module, (nn.Conv2d, nn.Linear, nn.Flatten)):
            snn.update([(str(i), quartz.Repeat(module))])
            i += 1

        elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
            snn.update(
                [
                    (
                        str(i),
                        quartz.layer.PoolingWrapper(
                            module=module, t_max=t_max
                        ),
                    )
                ]
            )
            i += 1

    return nn.Sequential(snn)
