import quartz
import torch
import torch.nn as nn


def test_conversion():
    ann = nn.Conv2d(2, 4, 3, bias=False)
    # ann = nn.Sequential(
    #     nn.Conv2d(2, 4, 3, bias=False),
    #     nn.ReLU(),
        # nn.Conv2d(4, 8, 3, bias=False),
        # nn.ReLU(),
        # nn.Flatten(),
        # nn.Linear(8, 10, bias=False),
    # )

    t_max = 2**8
    batch_size = 2

    def hook(module, input, output) -> torch.Tensor:
        return quartz.quantize_inputs(output, t_max)

    # for module in ann.modules():
    #     module.register_forward_hook(hook)

    values = torch.rand((batch_size, 2, 5, 5)) / 3
    q_values = quartz.quantize_inputs(values, t_max)

    ann_output = ann(q_values)
    q_ann_output = quartz.quantize_inputs(ann_output, t_max)

    snn = quartz.IF(t_max=t_max, rectification=False) # quartz.from_model(ann, t_max=t_max, batch_size=batch_size)
    temp_q_values = quartz.encode_inputs(q_values, t_max=t_max)
    ann_output = ann(temp_q_values.flatten(0, 1)).unflatten(0, (batch_size, -1))
    temp_output = snn(ann_output)
    snn_output = quartz.decode_outputs(temp_output, t_max=t_max)

    assert q_ann_output.shape == snn_output.shape
    torch.testing.assert_close(q_ann_output, snn_output, atol=0.05, rtol=0.2)
