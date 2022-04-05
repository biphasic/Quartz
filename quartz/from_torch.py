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
    for name, module in list(model.named_children()):
        # if it's one of the layers we're looking for, substitute it
        if isinstance(module, nn.ReLU):
            snn.update(
                [
                    (
                        name,
                        quartz.IFSqueeze(
                            t_max=t_max,
                            rectification=True,
                            bias=last_bias,
                            batch_size=batch_size,
                        ),
                    )
                ]
            )

        elif isinstance(module, (nn.Conv2d, nn.Linear, nn.Flatten)):
            snn.update([(name, module)])
            if hasattr(module, "bias") and module.bias is not None:
                last_bias = module.bias.clone()

        elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d,)):
            snn.update([(name, quartz.layer.PoolingWrapper(module, t_max=t_max))])

        # if in turn it has children, go iteratively inside
        elif len(list(module.named_children())):
            assert False
            snn.update([(name, quartz.IF(t_max=t_max, rectification=True))])

    return nn.Sequential(snn)
