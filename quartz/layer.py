import torch.nn as nn
import sinabs.layers as sl
import sinabs.activation as sina
import torch
from .utils import encode_inputs, decode_outputs
from dataclasses import dataclass


class IF(sl.StatefulLayer):
    def __init__(
        self,
        t_max: int,
        rectification: bool = True,
        record_v_mem: bool = False,
    ):
        super().__init__(state_names=["v_mem", "i_syn"])
        self.t_max = t_max
        self.rectification = rectification
        self.spike_threshold = t_max
        self.spike_fn = sina.SingleSpike
        self.reset_fn = sina.MembraneReset()
        self.surrogate_grad_fn = sina.Heaviside()
        self.record_v_mem = record_v_mem

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, n_time_steps, *trailing_dim = data.shape

        if not self.is_state_initialised() or not self.state_has_shape(
            (batch_size, *trailing_dim)
        ):
            self.init_state_with_shape((batch_size, *trailing_dim))

        readout_time = (self.t_max - 1) * 2
        rectification_time = (self.t_max - 1) * 3
        # counter weight and readout
        data[:, readout_time] += 1 - data.sum(1)

        output_spikes = torch.zeros_like(data)
        if self.record_v_mem:
            self.v_mem_recorded = torch.zeros_like(data)

        for step in range(n_time_steps):
            self.i_syn = self.i_syn + data[:, step]
            self.v_mem = self.v_mem + self.i_syn

            state = dict(self.named_buffers())
            input_tensors = [state[name] for name in self.spike_fn.required_states]
            spikes = self.spike_fn.apply(
                *input_tensors, self.spike_threshold, self.surrogate_grad_fn
            )
            state = self.reset_fn(spikes, state, self.spike_threshold)

            output_spikes[:, step] = spikes
            self.v_mem = state["v_mem"]
            # reset input current for neurons that spiked
            self.i_syn[spikes.bool()] = 0

            # if relu is activated, make sure neurons that haven't spiked until
            # now do so at t_max * 3
            if self.rectification and step == rectification_time:
                non_spiking_indices = list(torch.where(output_spikes.sum(1) == 0))
                if len(non_spiking_indices) == 0:
                    # every neuron spiked
                    break
                self.v_mem[non_spiking_indices] = 0
                self.i_syn[non_spiking_indices] = 0
                non_spiking_indices.insert(1, step)
                output_spikes[non_spiking_indices] = 1.0
                break

            if self.record_v_mem:
                self.v_mem_recorded[:, step] = self.v_mem

        # return shifted output_spikes
        return torch.hstack(
            (
                output_spikes[:, self.t_max - 1 :],
                torch.zeros_like(data)[:, : self.t_max - 1],
            )
        )

    def __repr__(self):
        return f"IF(t_max={self.t_max}, rectification={self.rectification})"


class Repeat(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.module(x.flatten(0, 1))
        x = x.unflatten(0, (batch_size, -1))

        if hasattr(self.module, 'bias') and self.module.bias is not None:
            batch_size, n_time_steps, *trailing_dim = x.shape
            bias = self.module.bias.clone().clamp(min=-1.5, max=1.5)
            while len(trailing_dim) != len(bias.shape):
                bias = bias.unsqueeze(1)
            x = x - bias
            t_max = (n_time_steps + 4) // 4
            bias_per_neuron = (
                torch.ones((batch_size, *trailing_dim), device=x.device) * bias
            )
            temp_bias = encode_inputs(bias_per_neuron, t_max=t_max)
            x += temp_bias
        return x


class IFSqueeze(IF, sl.SqueezeMixin):
    """
    Same as parent class, only takes in squeezed 4D input (Batch*Time, Channel, Height, Width)
    instead of 5D input (Batch, Time, Channel, Height, Width) in order to be compatible with
    layers that can only take a 4D input, such as convolutional and pooling layers.
    """

    def __init__(
        self,
        batch_size=None,
        num_timesteps=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.squeeze_init(batch_size, num_timesteps)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.squeeze_forward(input_data, super().forward)

    def __repr__(self):
        return "Squeeze" + super().__repr__()


class PoolingWrapper(nn.Module):
    def __init__(self, module, t_max):
        super().__init__()
        self.module = module
        self.t_max = t_max

    def forward(self, data):
        data = decode_outputs(data, t_max=self.t_max).to(data.device)
        data = self.module(data)
        return encode_inputs(data, t_max=self.t_max).to(data.device)


class PoolingWrapperSqueeze(PoolingWrapper, sl.SqueezeMixin):
    def __init__(
        self,
        batch_size=None,
        num_timesteps=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.squeeze_init(batch_size, num_timesteps)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.squeeze_forward(input_data, super().forward)


