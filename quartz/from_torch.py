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
    # iterate over the children
    for name, module in list(model.named_children()):
        # if it's one of the layers we're looking for, substitute it
        if isinstance(module, nn.ReLU):
            snn.update(
                [
                    (
                        name,
                        quartz.IFSqueeze(
                            t_max=t_max, rectification=True, batch_size=batch_size
                        ),
                    )
                ]
            )

        elif isinstance(module, (nn.Conv2d, nn.Linear, nn.AvgPool2d, nn.Flatten)):
            snn.update([(name, module)])

        # if in turn it has children, go iteratively inside
        elif len(list(module.named_children())):
            snn.update([(name, quartz.IF(t_max=t_max, rectification=True))])

    return nn.Sequential(snn)
