import torch
import quartz


def test_encoding_input():
    t_max = 2**4
    n_layers = 1
    values = torch.rand(2,3,4,5)
    input = quartz.encode_inputs(values, t_max=t_max, n_layers=n_layers)

    assert len(input.shape) == 5
    assert input.sum() == values.numel()