import torch
import torch.nn as nn
import quartz
import pytest


@pytest.mark.parametrize("input_dims", [(1, 1, 1, 1), (1, 1, 10, 10), (2, 3, 5, 5)])
def test_if_layer(input_dims):
    t_max = 2**5
    static_data = torch.rand(input_dims)
    q_input = quartz.quantize_inputs(static_data, t_max)
    input_data = quartz.encode_inputs(static_data, t_max)

    quartz_layer = quartz.IF(t_max=t_max, rectification=False)

    output_quartz_raw = quartz_layer(input_data)
    output_quartz = quartz.decode_outputs(output_quartz_raw, t_max)

    torch.testing.assert_close(q_input, output_quartz)


@pytest.mark.parametrize("input_dims", [(1, 1, 1, 1), (4, 3, 2, 2)])
def test_if_rectification(input_dims):
    t_max = 2**5
    static_data = torch.rand(input_dims) - 0.5
    input_data = quartz.encode_inputs(static_data, t_max)

    quartz_layer = quartz.IF(t_max=t_max, rectification=True)

    output_quartz_raw = quartz_layer(input_data)
    output_quartz = quartz.decode_outputs(output_quartz_raw, t_max)

    q_input = quartz.quantize_inputs(static_data, t_max)
    rect_q_input = torch.nn.functional.relu(q_input)

    assert torch.allclose(rect_q_input, output_quartz)


def test_conv_output():
    t_max = 2**8
    batch_size = 3
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


def test_linear_output():
    t_max = 2**8
    batch_size = 3
    values = (torch.rand(batch_size, 2000) - 0) / 3
    q_values = quartz.quantize_inputs(values, t_max)

    linear_layer = nn.Linear(2000, 2000, bias=False)
    ann_output = linear_layer(q_values)
    q_ann_output = quartz.quantize_inputs(ann_output, t_max=t_max)

    temp_q_values = quartz.encode_inputs(q_values, t_max=t_max)
    temp_linear = linear_layer(temp_q_values.flatten(0, 1)).unflatten(
        0, (batch_size, -1)
    )
    quartz_output = quartz.IF(t_max=t_max, rectification=False)(temp_linear)
    q_quartz_output = quartz.decode_outputs(quartz_output, t_max=t_max)

    torch.testing.assert_close(q_ann_output, q_quartz_output, atol=0.05, rtol=0.1)


def test_pooling_output():
    t_max = 2**8
    batch_size = 3

    values = torch.rand((batch_size, 2, 10, 10)) / 3
    q_values = quartz.quantize_inputs(values, t_max)

    pooling_layer = nn.MaxPool2d(2)
    quartz_layer = quartz.layer.PoolingWrapperSqueeze(
        module=pooling_layer, t_max=t_max, batch_size=batch_size
    )
    ann_output = pooling_layer(q_values)

    temp_q_values = quartz.encode_inputs(q_values, t_max=t_max)
    temp_pooling = quartz_layer(temp_q_values.flatten(0, 1)).unflatten(
        0, (batch_size, -1)
    )
    snn_output = quartz.decode_outputs(temp_pooling, t_max=t_max)

    assert ann_output.shape == snn_output.shape
    torch.testing.assert_close(ann_output, snn_output, atol=0.01, rtol=0.2)


def test_repeat():
    t_max = 2**3
    values = torch.rand((5, 2, 3, 3))
    q_values = quartz.quantize_inputs(values, t_max)
    temp_data = quartz.encode_inputs(q_values, t_max=t_max)

    module = nn.Conv2d(2, 4, 3)
    timed_module = quartz.Repeat(module)

    output = timed_module(temp_data)

    assert output.shape == (5, 28, 4, 1, 1)