import torch


def encode_inputs(data, t_max):
    time_index = ((t_max - 1) * (1 - data)).round().long().flatten()
    input = torch.zeros(data.shape[0], 3 * t_max - 3, *data.shape[1:], device=data.device)
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
    values = torch.zeros(batch_size, *trailing_dim, device=output.device)
    indices = list(torch.where(output == 1))
    time_values = (t_max - 1 - indices[1]) / (t_max - 1)
    indices.pop(1)
    values[indices] = time_values
    return values


def quantize_parameters(weights, biases, weight_acc, t_max):
    quantized_weights = (weight_acc * weights).round() / weight_acc
    quantized_biases = (biases * t_max).round() / t_max
    return quantized_weights, quantized_biases


def quantize_inputs(inputs, t_max):
    return (inputs * (t_max - 1)).round() / (t_max - 1)
