import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import sinabs.layers as sl


def encode_inputs(data, t_max):
    time_index = ((t_max - 1) * (2 - data)).round().long().flatten()
    input = torch.zeros(
        data.shape[0], 4 * t_max - 4, *data.shape[1:], device=data.device
    )
    if len(data.shape) == 4:
        batch, channel, height, width = data.shape
        batch_index = (
            torch.arange(batch).repeat_interleave(channel * height * width).repeat(1)
        )
        channel_index = (
            torch.arange(channel).repeat_interleave(height * width).repeat(batch)
        )
        height_index = (
            torch.arange(height).repeat_interleave(width).repeat(batch * channel)
        )
        width_index = torch.arange(width).repeat(batch * channel * height)

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
    time_values = (2 * t_max - 2 - indices[1]) / (t_max - 1)
    indices.pop(1)
    values[indices] = time_values
    return values


def quantize_parameters(weights, biases, weight_acc, t_max):
    quantized_weights = (weight_acc * weights).round() / weight_acc
    quantized_biases = (biases * t_max).round() / t_max
    return quantized_weights, quantized_biases


def quantize_inputs(inputs, t_max):
    return (inputs * (t_max - 1)).round() / (t_max - 1)


def get_accuracy(model, data_loader, device, t_max=None):
    correct_pred = 0
    n = 0
    model.eval()
    early_spikes = []
    for X, y_true in iter(data_loader):
        X = X.to(device)
        if t_max is not None:
            X = encode_inputs(X, t_max=t_max).to(device)
        y_true = y_true.to(device)
        with torch.no_grad():
            y_prob = model(X)
        if t_max is not None:
            y_prob = decode_outputs(y_prob, t_max=t_max)
            early_spikes.append([module.early_spikes for module in model.children() if isinstance(module, sl.StatefulLayer)])
        _, predicted_labels = torch.max(y_prob, 1)
        n += y_true.size(0)
        correct_pred += (predicted_labels == y_true).sum()
    if t_max is not None: print("Early spike % / layer: ", 100*torch.tensor(early_spikes).mean(0))
    return (correct_pred.float() / n).item() * 100


def plot_output_histograms(model, sample_input, output_layers, t_max=None):
    fig, axes = plt.subplots(len(output_layers), 1, figsize=(4, int(len(output_layers)*2)))
    if not isinstance(axes, list):
        axes = [axes]
    model.eval()

    activations = []
    def hook(module, inp, output):
        activations.append(output.detach()) # 

    for i, layer in enumerate(output_layers):
        handle = layer.register_forward_hook(hook)

        with torch.no_grad():
            model(sample_input)

        if t_max is not None:
            output = decode_outputs(activations[0], t_max=t_max)
        else:
            output = activations[0]
        
        axes[i].hist(output.cpu().ravel().numpy(), bins=30)
        activations = []
        handle.remove()