import torch
import quartz


def test_encoding_input():
    t_max = 2 ** 4
    values = torch.rand(2, 3, 4, 5)
    input = quartz.encode_inputs(values, t_max=t_max)

    assert len(input.shape) == 5
    assert input.sum() == values.numel()


def test_decoding_output():
    t_max = 2 ** 4
    values = torch.rand(2, 3, 1, 1)
    q_values = quartz.quantize_inputs(values, t_max)
    input = quartz.encode_inputs(values, t_max=t_max)
    # shift spikes by 1 t_max
    new_input = torch.hstack(
        (torch.zeros_like(input)[:, : (t_max - 1)], input[:, : 2 * t_max + 1])
    )
    output = quartz.decode_outputs(new_input, t_max=t_max)
    assert (output == q_values).all()
