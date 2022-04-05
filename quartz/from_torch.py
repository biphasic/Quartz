import quartz
import copy
import torch
import torch.nn as nn
from collections import OrderedDict


def from_model(
    model: nn.Module,
    t_max: int,
    batch_size: int,
):
    # model = copy.deepcopy(model)

    snn = OrderedDict()
    last_bias = 0
    # iterate over the children
    i = 0
    for name, module in list(model.named_children()):
        # if it's one of the layers we're looking for, substitute it
        if isinstance(module, nn.ReLU):
            snn.update(
                [
                    (
                        str(i),
                        quartz.IFSqueeze(
                            t_max=t_max,
                            rectification=True,
                            bias=last_bias,
                            batch_size=batch_size,
                        ),
                    )
                ]
            )
            i += 1

        elif isinstance(module, (nn.Conv2d, nn.Linear, nn.Flatten, nn.Dropout, nn.Dropout2d)):
            snn.update([(str(i), module)])
            if hasattr(module, "bias") and module.bias is not None:
                last_bias = module.bias.clone()
            i += 1

        elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
            snn.update(
                [
                    (
                        str(i),
                        quartz.layer.PoolingWrapperSqueeze(
                            module=module, t_max=t_max, batch_size=batch_size
                        ),
                    )
                ]
            )
            i += 1

    return nn.Sequential(snn)
