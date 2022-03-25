import quartz
import torch
import torch.nn as nn


def test_conversion():
    torch.set_printoptions(sci_mode=False)

    # ann = nn.Conv2d(2, 4, 3, bias=False)
    ann = nn.Sequential(
        nn.Conv2d(2, 4, 3, bias=False),
        nn.ReLU(),
        nn.Conv2d(4, 8, 3, bias=False),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8, 10, bias=False),
        nn.ReLU(),
    )

    t_max = 2**8 + 1
    batch_size = 2

    values = torch.rand((batch_size, 2, 5, 5)) / 3
    q_values = quartz.quantize_inputs(values, t_max)
    ann_output = ann(q_values)

    snn = quartz.from_model(ann, t_max=t_max, batch_size=batch_size)
    temp_q_values = quartz.encode_inputs(q_values, t_max=t_max)
    temp_output = snn(temp_q_values.flatten(0, 1)).unflatten(0, (batch_size, -1))
    snn_output = quartz.decode_outputs(temp_output, t_max=t_max)

    assert ann_output.shape == snn_output.shape
    print(ann_output - snn_output)
    torch.testing.assert_close(ann_output, snn_output, atol=0.05, rtol=0.2)
