import sinabs.layers as sl
import sinabs.activation as sina
import torch
import torch.nn as nn


class IF(sl.StatefulLayer):
    def __init__(
        self,
        t_max: int,
        index: int,
        rectification: bool = True,
        record_v_mem: bool = False,
    ):
        super().__init__(state_names=["v_mem", "i_syn"])
        self.t_max = t_max
        self.index = index
        self.rectification = rectification
        self.act_fn = sina.ActivationFunction(spike_threshold=t_max, spike_fn=sina.SingleSpike, reset_fn=sina.MembraneReset())
        self.record_v_mem = record_v_mem

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, n_time_steps, *trailing_dim = data.shape

        if not self.is_state_initialised() or not self.state_has_shape(
            (batch_size, *trailing_dim)
        ):
            self.init_state_with_shape((batch_size, *trailing_dim))

        counter_weight = torch.zeros_like(data)
        counter_weight[:, self.t_max*(self.index+1)] = 1 - data.sum()
        data = data + counter_weight

        output_spikes = torch.zeros_like(data)
        if self.record_v_mem: self.v_mem_recorded = torch.zeros_like(data)

        for step in range(n_time_steps):
            self.i_syn = self.i_syn + data[:, step]
            self.v_mem = self.v_mem + self.i_syn

            spikes, state = self.act_fn(dict(self.named_buffers()))
            output_spikes[:, step] = spikes
            self.v_mem = state["v_mem"]
            self.i_syn[spikes.bool()] = 0

            if self.rectification and step == self.t_max*2 - 1:
                batch, *trailing_dim = torch.where(output_spikes.sum(1) == 0)
                if batch.numel() == 0: break
                output_spikes[batch, step, trailing_dim] = 1.
                self.v_mem[batch, trailing_dim] = 0
                self.i_syn[batch, trailing_dim] = 0
                break

            if self.record_v_mem: self.v_mem_recorded[:, step] = self.v_mem
        
        return output_spikes


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
