import torch


def encode_inputs(data, t_max):
    time_index = ((t_max - 1) * (1 - data)).round().long().flatten()
    input = torch.zeros(data.shape[0], 3 * t_max - 3, *data.shape[1:])
    if len(data.shape) == 4:
        batch, channel, height, width = data.shape
        batch_index = torch.arange(batch).repeat_interleave(channel*height*width).repeat(1)
        channel_index = torch.arange(channel).repeat_interleave(height*width).repeat(batch)
        height_index = torch.arange(height).repeat_interleave(width).repeat(batch*channel)
        width_index = torch.arange(width).repeat(batch*channel*height)

        input[batch_index, time_index, channel_index, height_index, width_index] = 1.0
    
    elif len(data.shape) == 2:
        batch, channel = data.shape
        batch_index = torch.arange(batch).repeat_interleave(channel)
        channel_index = torch.arange(channel).repeat(batch)

        input[batch_index, time_index, channel_index] = 1.0
    return input


def decode_outputs(output, t_max):
    batch_size, n_time_steps, *trailing_dim = output.shape
    values = torch.zeros(batch_size, *trailing_dim)
    indices = [torch.where(output == 1)]
    if len(indices[0]) == 5:
        for b, t, c, h, w in indices:
            values[b, c, h, w] = (t_max - 1 - t) / (t_max - 1)
    elif len(indices[0]) == 3:
        for b, t, c in indices:
            values[b, c] = (t_max - 1 - t) / (t_max - 1)
    return values


def quantize_parameters(weights, biases, weight_acc, t_max):
    quantized_weights = (weight_acc * weights).round() / weight_acc
    quantized_biases = (biases * t_max).round() / t_max
    return quantized_weights, quantized_biases


def quantize_inputs(inputs, t_max):
    return (inputs * (t_max - 1)).round() / (t_max - 1)
