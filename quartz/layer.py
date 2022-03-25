import sinabs.layers as sl
import sinabs.activation as sina
import torch
import torch.nn as nn


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

        # counter weight and readout
        data[:, self.t_max - 1] += 1 - data.sum(1)

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
            # now do so at t_max * 2
            if self.rectification and step == (self.t_max - 1) * 2:
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

        # return output_spikes
        return torch.hstack(
            (
                output_spikes[:, self.t_max - 1 :],
                torch.zeros_like(data)[:, : self.t_max - 1],
            )
        )


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
