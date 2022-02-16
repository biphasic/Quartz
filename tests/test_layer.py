import torch
import torch.nn as nn
import quartz
import pytest


@pytest.mark.parametrize("input_dims", [(1, 1, 1, 1), (1, 1, 10, 10)])
def test_if_layer(input_dims):
    t_max = 2 ** 5 + 1
    static_data = torch.rand(input_dims)
    q_input = quartz.quantize_inputs(static_data, t_max)
    input_data = quartz.encode_inputs(static_data, t_max)

    quartz_layer = quartz.IF(t_max=t_max, rectification=False)

    output_quartz_raw = quartz_layer(input_data)
    output_quartz = quartz.decode_outputs(output_quartz_raw, t_max)

    assert torch.allclose(q_input, output_quartz)


@pytest.mark.parametrize("input_dims", [(1, 1, 1, 1), (4, 3, 2, 2)])
def test_if_rectification(input_dims):
    t_max = 2 ** 5
    static_data = torch.rand(input_dims) - 0.5
    input_data = quartz.encode_inputs(static_data, t_max)

    quartz_layer = quartz.IF(t_max=t_max, rectification=True)

    output_quartz_raw = quartz_layer(input_data)
    output_quartz = quartz.decode_outputs(output_quartz_raw, t_max)

    q_input = quartz.quantize_inputs(static_data, t_max)
    rect_q_input = torch.nn.functional.relu(q_input)

    assert torch.allclose(rect_q_input, output_quartz)
