import quartz
import torch
import torch.nn as nn


def test_conversion():
    ann = nn.Sequential(
        nn.Conv2d(2, 4, 3, bias=False),
        nn.ReLU(),
        # nn.Conv2d(4, 8, 3, bias=False),
        # nn.ReLU(),
        # nn.Flatten(),
        # nn.Linear(8, 10, bias=False),
    )

    with torch.no_grad():
        for param in ann.parameters():
            param /= 3

    t_max = 2**8
    batch_size = 10

    def hook(module, input, output) -> torch.Tensor:
        return quartz.quantize_inputs(output, t_max)

    # for module in ann.modules():
    #     module.register_forward_hook(hook)

    snn = quartz.from_model(ann, t_max=t_max, batch_size=batch_size)

    static_input = torch.rand((batch_size, 2, 5, 5)) / 3
    q_static_input = quartz.quantize_inputs(static_input, t_max)
    temp_input = quartz.encode_inputs(static_input, t_max=t_max)

    static_output = ann(q_static_input)
    temp_output = snn(temp_input.flatten(0, 1)).unflatten(0, (batch_size, -1))
    snn_output = quartz.decode_outputs(temp_output, t_max=t_max)

    assert static_output.shape == snn_output.shape
    torch.testing.assert_close(static_output, snn_output, atol=0.05, rtol=0.2)
    # assert torch.allclose(snn_output, static_output)