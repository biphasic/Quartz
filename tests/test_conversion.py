import quartz
import torch
import torch.nn as nn


def test_conversion():
    torch.set_printoptions(sci_mode=False)

    ann = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(6, 12, kernel_size=5),
        nn.ReLU(),
        nn.Dropout2d(0.4),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(12, 120, kernel_size=4),
        nn.ReLU(),
        nn.Dropout2d(0.4),
        nn.Flatten(),
        nn.Linear(120, 10),
        nn.ReLU(),
    )

    ann.eval()

    t_max = 2**6 + 1
    batch_size = 2

    values = torch.rand((batch_size, 1, 28, 28)) / 3
    q_values = quartz.quantize_inputs(values, t_max)
    ann_output = ann(q_values)

    snn = quartz.from_model(ann, t_max=t_max)
    temp_q_values = quartz.encode_inputs(q_values, t_max=t_max)
    temp_output = snn(temp_q_values)
    snn_output = quartz.decode_outputs(temp_output, t_max=t_max)

    assert ann_output.shape == snn_output.shape
    print(
        f"Sum of ann output is {ann_output.sum()}, biggest difference between outputs is {(ann_output - snn_output).max()}"
    )
    torch.testing.assert_close(ann_output, snn_output, atol=0.05, rtol=0.1)
