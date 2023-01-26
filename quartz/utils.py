import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import sinabs.layers as sl
import numpy as np
from typing import List
import torch.nn as nn
import seaborn as sns
from torch.nn.utils.fusion import fuse_conv_bn_eval


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


def get_accuracy(model, data_loader, device, preprocess=None, calculate_early_spikes=False, calculate_output_time=False, t_max=None):
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
            if preprocess is not None:
                preprocess = preprocess.to(device)
                with torch.no_grad():
                    X = preprocess(X)
                    # cut off inputs that are not covered by preprocessing
                    X = torch.clamp(X, min=-1.9, max=2.)
            if t_max is not None:
                X = encode_inputs(X, t_max=t_max).to(device)
            y_true = y_true.to(device)
            with torch.no_grad():
                y_prob = model(X)
            if t_max is not None:
                if calculate_output_time:
                    earliest_output_spikes.append(t_max - torch.where(y_prob)[1].float().mean())
                y_prob = decode_outputs(y_prob, t_max=t_max)
                if calculate_early_spikes:
                    early_spikes.append([module.early_spikes for module in model.modules() if isinstance(module, sl.StatefulLayer)])
            predicted_labels = y_prob.argmax(1)
            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()
            progress_bar.set_postfix({'Valid_acc': (correct_pred.float() / n).item() * 100})
            progress_bar.update()
    metrics = {}
    if t_max is not None:
        metrics[t_max] = {
            'acc': (correct_pred.float() / n).item() * 100
        }
        if calculate_output_time:
            metrics[t_max]['time steps'] = round((n_spike_layers-1)*t_max + torch.tensor(earliest_output_spikes).mean().item())
        if calculate_early_spikes:
            metrics[t_max]['early spike per layer'] = 100*torch.tensor(early_spikes).mean(0).cpu().numpy().round(3)
            metrics[t_max]['early spikes'] = round(100*torch.tensor(early_spikes).mean().item(), 2)
    else:
        metrics['acc'] = (correct_pred.float() / n).item() * 100
    return metrics


def plot_output_comparison(model1, model2, sample_input, output_layers, every_n=1, every_c=1, savefig=None):
    fig, axes = plt.subplots(len(output_layers), 1, figsize=(6, int(len(output_layers)*3)))
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
        
        # data1 = activations1[-1].cpu().ravel().numpy()[::every_n]
        # data2 = activations2[-1].cpu().ravel().numpy()[::every_n]
        # sorted_idx = np.argsort(data1)
        # data1_sorted = data1[sorted_idx]
        # data2_sorted = data2[sorted_idx]
        # axes[i].scatter(data1, data2)

        data1 = torch.moveaxis(activations1[-1].cpu(), 1, 0).flatten(1, -1).numpy()[::every_c, ::every_n]
        data2 = torch.moveaxis(activations2[-1].cpu(), 1, 0).flatten(1, -1).numpy()[::every_c, ::every_n]
        # sorted_idx = np.argsort(data1, axis=1)
        # data1_sorted = data1.ravel()[sorted_idx.ravel()].reshape(data1.shape[0], -1)[::every_c, ::every_n]
        # data2_sorted = data2.ravel()[sorted_idx.ravel()].reshape(data2.shape[0], -1)[::every_c, ::every_n]

        # print(data1.shape)
        for j in range(data1.shape[0]):
            axes[i].scatter(data1[j], data2[j])
        
        axes[i].set_xlabel(f"Original activations layer {layer}")
        axes[i].set_ylabel('Normalised activations')
        axes[i].grid(True)
        # axes[i].legend()
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


def normalize_weights(
    ann: nn.Module,
    sample_data: torch.Tensor,
    param_layer_names: List[str],
    percentile: float = 99,
):
    """
    Rescale the weights of the network, such that the activity of each specified layer is normalized.

    Args:
         ann(nn.Module): Torch module
         sample_data (nn.Tensor): Input data to normalize the network with
         param_layers (List[str]): List of names of layers to verify activity of normalization. 
         percentile (float): A number between 0 and 100 to determine activity to be normalized by.
          where a 100 corresponds to the max activity of the network. Defaults to 99.
    """
    max_outputs = []
    def save_data(lyr, input, output):
        max_outputs.append(np.percentile(output.cpu().numpy(), percentile))

    named_layers = dict(ann.named_children())

    for i in range(len(param_layer_names)):
        param_layer = named_layers[param_layer_names[i]]

        handle = param_layer.register_forward_hook(save_data)

        with torch.no_grad():
            _ = ann(sample_data)

            max_layer_output = max_outputs[-1]

            param_layer.weight.data /= max_layer_output
            if hasattr(param_layer, 'bias'):
                bias_scale = np.product(np.array(max_outputs))
                # print(f"weight scale: {1/max_layer_output}, bias_scale: {1/bias_scale}")
                param_layer.bias.data /= bias_scale

            # # Rescale weights to normalize max output
            # for p in param_layer.parameters():
            #     p.data *= 1 / max_lyr_out

        handle.remove()


def normalize_weights_alternative(
    ann: nn.Module,
    sample_data: torch.Tensor,
    param_layer_names,
    percentile: float = 99,
):
    ann = ann.eval()
    max_outputs = []
    def save_data(lyr, input, output):
        max_outputs.append(np.percentile(output.cpu().detach().numpy(), percentile))

    named_layers = dict(ann.named_children())
    param_layers = [named_layers[param_layer_name] for param_layer_name in param_layer_names]
    handles = []
    for param_layer in param_layers:
        handle = param_layer.register_forward_hook(save_data)
        handles.append(handle)

    with torch.no_grad():
        _ = ann(sample_data)

    prev_scale = 1
    for i, param_layer in enumerate(param_layers):
        prev_output = 1 if i == 0 else max_outputs[i-1]
        weight_scale = prev_output/max_outputs[i]
        param_layer.weight.data *= weight_scale
        bias_scale = prev_scale * weight_scale
        param_layer.bias.data *= bias_scale
        print(f"weight scale: {weight_scale}, bias_scale: {bias_scale}")
        prev_scale = weight_scale

    [handle.remove() for handle in handles]


def plot_output_comparison_ann_snn(ann, snn, sample_input, ann_output_layers, snn_output_layers, t_max, every_n=1, every_c=1, savefig=None):
    assert len(ann_output_layers) == len(snn_output_layers)
    fig, axes = plt.subplots(len(ann_output_layers), 1, figsize=(6, int(len(ann_output_layers)*3)))
    if savefig: plt.suptitle(f"t_max: {t_max}")
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    ann = ann.eval()
    snn = snn.eval()

    ann_layers = dict(ann.named_children())
    snn_layers = dict(snn.named_children())

    activations1 = []
    activations2 = []
    def hook1(module, inp, output):
        activations1.append(output.detach())

    def hook2(module, inp, output):
        activations2.append(decode_outputs(output.detach(), t_max=t_max))

    for i, (ann_layer_name, snn_layer_name) in enumerate(list(zip(ann_output_layers, snn_output_layers))):
        ann_layer = ann_layers[ann_layer_name]
        snn_layer = snn_layers[snn_layer_name]

        handle1 = ann_layer.register_forward_hook(hook1)
        handle2 = snn_layer.register_forward_hook(hook2)

        ann(sample_input)
        snn(encode_inputs(sample_input, t_max=t_max).to(sample_input.device))

        # data1 = torch.moveaxis(activations1[-1].cpu(), 1, 0).flatten(1, -1).numpy()[::every_c, ::every_n]
        # data2 = torch.moveaxis(activations2[-1].cpu(), 1, 0).flatten(1, -1).numpy()[::every_c, ::every_n]
        data1 = torch.moveaxis(activations1[-1].cpu(), 1, 0).flatten().numpy()[::every_n]
        data2 = torch.moveaxis(activations2[-1].cpu(), 1, 0).flatten().numpy()[::every_n]

        # sorted_idx = np.argsort(data1)
        # data1_sorted = data1[sorted_idx]
        # data2_sorted = data2[sorted_idx]
        
        # sns.histplot(x=data1, y=data2, bins=20, cbar=True, cbar_kws=dict(shrink=.75), ax=axes[i])
        axes[i].scatter(data1, data2)

        # axis = fig.add_subplot(len(ann_output_layers), 1, i+1)#, projection='scatter_density')
        # axis.scatter(data1, data2, c=)#, c=data1_sorted[::-1])
        
        axes[i].set_xlabel(f"ANN activations layer {ann_layer_name}")
        axes[i].set_ylabel('SNN activations')
        # axis[i].grid(True)
        # axes[i].legend()
        activations1 = []
        activations2 = []
        handle1.remove()
        handle2.remove()
    if savefig:
        plt.tight_layout()
        plt.savefig(savefig)


def count_n_neurons(model, sample_input, add_last_layer=False):
    assert sample_input.shape[0] == 1
    n_output_neurons = []
    handles = []
    def count_neurons(self, input, output):
        n_output_neurons.append(output.numel())
    for layer in model.modules():
        if isinstance(layer, (nn.ReLU, nn.ReLU6)):
            handles.append(layer.register_forward_hook(count_neurons))
    with torch.no_grad():
        output = model(sample_input)
    n_neurons = torch.tensor(n_output_neurons).sum().item()
    if add_last_layer:
        n_neurons += output.numel()
    [handle.remove() for handle in handles]
    return n_neurons


def remove_identity_layers(model):
    children_list = list(model.named_children())
    for name, module in children_list:
        if list(module.named_children()):
            remove_identity_layers(module)
        if isinstance(module, nn.Identity):
            delattr(model, name)


def fuse_all_conv_bn(model):
    """
    Fuses all consecutive Conv2d and BatchNorm2d layers.
    License: Copyright Zeeshan Khan Suri, CC BY-NC 4.0
    """
    stack = []
    for name, module in model.named_children(): # immediate children
        if list(module.named_children()): # is not empty (not a leaf)
            fuse_all_conv_bn(module)
            
        if isinstance(module, nn.BatchNorm2d):
            if isinstance(stack[-1][1], nn.Conv2d):
                setattr(model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module))
                setattr(model, name, nn.Identity())
        else:
            stack.append((name, module))


def plot_output_comparison_new(model1, model2, sample_input, every_n=1, every_c=1, savefig=None):
    sns.set_theme(style="dark")
    output_layer_pairs = [((name1, layer1), (name2, layer2)) for (name1, layer1), (name2, layer2) in zip(model1.named_modules(), model2.named_modules()) if isinstance(layer1, (nn.Conv2d, nn.Linear)) and isinstance(layer2, (nn.Conv2d, nn.Linear))]
    n_output_layers = len(output_layer_pairs)
    fig, axes = plt.subplots(n_output_layers, 1, figsize=(6, int(n_output_layers*4)))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    model1 = model1.eval()
    model2 = model2.eval()

    activations1 = []
    activations2 = []
    def hook1(module, inp, output):
        activations1.append(output.detach())

    def hook2(module, inp, output):
        activations2.append(output.detach())

    for i, ((name1, layer1), (name2, layer2)) in enumerate(output_layer_pairs):
        if isinstance(layer1, (nn.Conv2d, nn.Linear)):
            handle1 = layer1.register_forward_hook(hook1)
            handle2 = layer2.register_forward_hook(hook2)

            model1(sample_input)
            model2(sample_input)

            data1 = torch.moveaxis(activations1[-1].cpu(), 1, 0).flatten(1, -1).numpy()[::every_c, ::every_n]
            data2 = torch.moveaxis(activations2[-1].cpu(), 1, 0).flatten(1, -1).numpy()[::every_c, ::every_n]

            # print(data1.shape)
            sns.scatterplot(x=data1.ravel(), y=data2.ravel(), s=10, color=".15", ax=axes[i])
            sns.histplot(x=data1.ravel(), y=data2.ravel(), bins=100, pthresh=.1, cmap="mako", ax=axes[i])
            
            axes[i].set_xlabel(f"Original activations layer {name1}")
            axes[i].set_ylabel('Normalised activations')
            axes[i].grid(True)
            activations1 = []
            activations2 = []
            handle1.remove()
            handle2.remove()
    if savefig:
        plt.tight_layout()
        plt.savefig(savefig)


def normalize_outputs(
    model: nn.Module,
    sample_data: torch.Tensor,
    percentile: float,
    max_outputs = []
):
    def save_data(lyr, input, output):
        max_outputs.append(np.percentile(output.cpu().numpy(), percentile))

    module_input = []
    def get_module_input(module, input, output):
        module_input.append(input[0])

    for name, module in model.named_children(): # immediate children
        if list(module.named_children()): # is not empty (not a leaf)
            handle = module.register_forward_hook(get_module_input)
            with torch.no_grad():
                model(sample_data)
            sample_input = module_input[-1]
            max_outputs = normalize_outputs(module, sample_input, percentile, max_outputs)
            handle.remove()

        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handle = module.register_forward_hook(save_data)

            with torch.no_grad():
                _ = model(sample_data)
                max_layer_output = max_outputs[-1]
                module.weight.data /= max_layer_output
                if hasattr(module, 'bias'):
                    bias_scale = np.product(np.array(max_outputs))
                    # print(f"weight scale: {1/max_layer_output}, bias_scale: {1/bias_scale}")
                    module.bias.data /= bias_scale
            handle.remove()
    return max_outputs