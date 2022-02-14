import torch


def encode_inputs(data, t_max, n_layers):
    assert len(data.shape) == 4
    input = torch.zeros(data.shape[0], (n_layers+1)*t_max-n_layers, *data.shape[1:])
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            for h in range(data.shape[2]):
                for w in range(data.shape[3]):
                    input[b, (t_max*(1-data[b, c, h, w])).round().long(), c, h, w] = 1.
    return input

def quantize_parameters(weights, biases, weight_acc, t_max):
    quantized_weights = (weight_acc*weights).round()/weight_acc
    quantized_biases = (biases*t_max).round()/t_max
    return quantized_weights, quantized_biases

def quantize_inputs(inputs, t_max):
    return (inputs*t_max).round()/t_max