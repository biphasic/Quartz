import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import sinabs.layers as sl
import numpy as np


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
    model = model.to(device)
    model.eval()
    n_spike_layers = sum([isinstance(layer, sl.StatefulLayer) for layer in model.modules()])
    early_spikes = []
    earliest_output_spikes = []
    with tqdm(total=len(data_loader)) as progress_bar:
        for X, y_true in iter(data_loader):
            X = X.to(device)
            if t_max is not None:
                X = encode_inputs(X, t_max=t_max).to(device)
            y_true = y_true.to(device)
            with torch.no_grad():
                y_prob = model(X)
            if t_max is not None:
                # earliest_output_spikes.append(t_max - torch.where(y_prob)[1].float().mean())
                y_prob = decode_outputs(y_prob, t_max=t_max)
                # early_spikes.append([module.early_spikes for module in model.children() if isinstance(module, sl.StatefulLayer)])
            predicted_labels = y_prob.argmax(1)
            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()
            progress_bar.set_postfix({'Valid_acc': (correct_pred.float() / n).item() * 100})
            progress_bar.update()
        # normally n_spike_layers - 1 + the earliest spikes of the last layer but also need to add input latency.
        # if t_max is not None: print(f"Earliest spike at {(n_spike_layers)*t_max + torch.tensor(earliest_output_spikes).mean()} time steps.")
        # if t_max is not None: print("Early spike % / layer: ", 100*torch.tensor(early_spikes).mean(0))
    return (correct_pred.float() / n).item() * 100


def plot_output_comparison(model1, model2, sample_input, output_layers, every_n=1, savefig=None):
    fig, axes = plt.subplots(len(output_layers), 1, figsize=(4, int(len(output_layers)*3)))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    model1 = model1.eval()
    model2 = model2.eval()

    named_layers_model1 = dict(model1.named_children())
    named_layers_model2 = dict(model2.named_children())

    activations1 = []
    activations2 = []
    def hook1(module, inp, output):
        activations1.append(output.detach())

    def hook2(module, inp, output):
        activations2.append(output.detach())

    for i, layer in enumerate(output_layers):
        output_layer1 = named_layers_model1[layer]
        output_layer2 = named_layers_model2[layer]

        handle1 = output_layer1.register_forward_hook(hook1)
        handle2 = output_layer2.register_forward_hook(hook2)

        model1(sample_input)
        model2(sample_input)
        
        data1 = activations1[-1].cpu().ravel().numpy()
        data2 = activations2[-1].cpu().ravel().numpy()
        sorted_idx = np.argsort(data1)
        data1_sorted = data1[sorted_idx][::every_n]
        data2_sorted = data2[sorted_idx][::every_n]

        axes[i].scatter(data1_sorted, data2_sorted, label='Output corr')
        # axes[i].plot([data1_sorted[0], data1_sorted[-1]], [data1_sorted[0], data1_sorted[-1]], label='1:1 corr', color='C2')
        axes[i].set_xlabel(f"Original activations layer {layer}")
        axes[i].set_ylabel('Normalised activations')
        axes[i].grid(True)
        axes[i].legend()
        activations1 = []
        activations2 = []
        handle1.remove()
        handle2.remove()
    if savefig:
        plt.tight_layout()
        plt.savefig(savefig)


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