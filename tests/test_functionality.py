import torch
import torch.nn as nn
import quartz
import pytest


@pytest.mark.parametrize(
    "weight, value",
    [
        (
            1.0,
            0,
        ),
        (
            0.5,
            0,
        ),
        (
            1.0,
            0.2093,
        ),
        (
            0.2093,
            0.2093,
        ),
        (
            0.456,
            0.456,
        ),
        (
            -1.0,
            0.2093,
        ),
        (
            1.0,
            1,
        ),
        (
            0.5,
            1,
        ),
        (
            0,
            1,
        ),
        (
            1.0,
            1.25,
        ),
    ],
)
def test_inputs(weight, value):
    t_max = 2**8 + 1
    values = torch.ones((1, 1)) * value
    q_values = quartz.quantize_inputs(values, t_max)

    linear_layer = nn.Linear(1, 1, bias=False)
    linear_layer.weight = torch.nn.Parameter(
        torch.ones_like(linear_layer.weight) * weight
    )
    ann_output = linear_layer(q_values)
    q_ann_output = quartz.quantize_inputs(ann_output, t_max)

    temp_q_values = quartz.encode_inputs(q_values, t_max=t_max)
    temp_linear = linear_layer(temp_q_values.flatten(0, 1)).unflatten(0, (1, -1))
    quartz_output = quartz.IF(t_max=t_max, rectification=False)(temp_linear)
    q_quartz_output = quartz.decode_outputs(quartz_output, t_max=t_max)

    print(
        f"Ann output is {q_ann_output.item()}, snn output is {q_quartz_output.item()}."
    )

    assert torch.all(q_ann_output == q_quartz_output)


@pytest.mark.parametrize("bias", [0.234, 0.5, 0.55, 0.8, 1])
def test_bias(bias):
    t_max = 2**8 + 1
    values = torch.zeros((1, 1))
    q_bias = quartz.quantize_inputs(torch.tensor(bias).float(), t_max=t_max)

    linear_layer = nn.Linear(1, 1)
    linear_layer.weight = torch.nn.Parameter(torch.zeros_like(linear_layer.weight))
    linear_layer.bias = torch.nn.Parameter(torch.ones_like(linear_layer.bias) * q_bias)
    ann_output = linear_layer(values)
    q_ann_output = quartz.quantize_inputs(ann_output, t_max)

    temp_q_values = quartz.encode_inputs(values, t_max=t_max)
    temp_linear_layer = quartz.Repeat(linear_layer)
    temp_linear = temp_linear_layer(temp_q_values)
    quartz_output = quartz.IF(t_max=t_max, rectification=False)(temp_linear)
    q_quartz_output = quartz.decode_outputs(quartz_output, t_max=t_max)

    assert torch.all(q_ann_output == q_quartz_output)


def test_multi_bias():
    t_max = 2**8 + 1
    values = torch.zeros((1, 5, 1, 1))

    conv_layer = nn.Conv2d(5, 5, 1)
    conv_layer.weight = torch.nn.Parameter(torch.zeros_like(conv_layer.weight))
    conv_layer.bias.data = quartz.quantize_inputs(conv_layer.bias.data.clone(), t_max)
    # print(f"layer bias: {conv_layer.bias}")
    ann_output = conv_layer(values)
    q_ann_output = quartz.quantize_inputs(ann_output, t_max)

    temp_q_values = quartz.encode_inputs(values, t_max=t_max)
    temp_conv_layer = quartz.Repeat(conv_layer)
    temp_conv = temp_conv_layer(temp_q_values)
    quartz_output = quartz.IF(t_max=t_max, rectification=False)(temp_conv)
    q_quartz_output = quartz.decode_outputs(quartz_output, t_max=t_max)

    assert torch.all(q_ann_output == q_quartz_output)
