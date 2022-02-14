import torch
import torch.nn as nn
import quartz


def test_conv_output_equality():
    t_max = 2**8
    static_data = torch.rand(1,1,10,10)
    input_data = quartz.encode_inputs(static_data, t_max, 1)

    torch_layer = 

