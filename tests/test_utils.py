import torch
import torch.nn as nn
import quartz


def test_encoding_input():
    t_max = 2**4
    values = torch.rand(2, 3, 4, 5)
    input = quartz.encode_inputs(values, t_max=t_max)

    assert len(input.shape) == 5
    assert input.sum() == values.numel()


def test_decoding_output():
    t_max = 2**4
    values = torch.rand(2, 3, 1, 1) * 2
    q_values = quartz.quantize_inputs(values, t_max)
    temp_input = quartz.encode_inputs(values, t_max=t_max)

    output = quartz.decode_outputs(temp_input, t_max=t_max)
    assert (output == q_values).all()
