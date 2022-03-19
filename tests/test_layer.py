import torch
import torch.nn as nn
import quartz
import pytest


@pytest.mark.parametrize("input_dims", [(1, 1, 1, 1), (1, 1, 10, 10), (2, 3, 5, 5)])
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


def test_decoding_conv_output():
    t_max = 2 ** 8
    batch_size = 1
    values = (torch.rand(batch_size, 2, 5, 5) - 0) / 3
    q_values = quartz.quantize_inputs(values, t_max) 

    conv_layer = nn.Conv2d(2, 4, 3, bias=False)
    ann_output = conv_layer(q_values)
    q_ann_output = quartz.quantize_inputs(ann_output, t_max=t_max)

    temp_q_values = quartz.encode_inputs(q_values, t_max=t_max)
    temp_conv = conv_layer(temp_q_values.flatten(0, 1)).unflatten(0, (batch_size, -1))
    quartz_output = quartz.IF(t_max=t_max, rectification=False)(temp_conv)
    q_quartz_output = quartz.decode_outputs(quartz_output, t_max=t_max)

    torch.testing.assert_close(q_ann_output, q_quartz_output, atol=0.05, rtol=0.1)
