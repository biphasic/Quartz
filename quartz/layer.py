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

        self.v_mem_recorded = []

        if not self.is_state_initialised() or not self.state_has_shape(
            (batch_size, *trailing_dim)
        ):
            self.init_state_with_shape((batch_size, *trailing_dim))

        counter_weight = torch.zeros_like(data)
        counter_weight[:, self.t_max*(self.index+1)] = 1 - data.sum()
        data = data + counter_weight

        output_spikes = []
        for step in range(n_time_steps):
            self.i_syn = self.i_syn + data[:, step]
            self.v_mem = self.v_mem + self.i_syn

            spikes, state = self.act_fn(dict(self.named_buffers()))
            output_spikes.append(spikes)
            self.v_mem = state["v_mem"]
            self.i_syn[spikes.bool()] = 0

            if self.rectification: 
            if self.record_v_mem: self.v_mem_recorded.append(self.v_mem)
        
        if self.record_v_mem: self.v_mem_recorded = torch.stack(self.v_mem_recorded, 1)

        spikes = torch.stack(output_spikes, 1)
        if self.rectification and (spikes[:, :self.t_max*(self.index+2)].sum(1) == 0).any():
            spikes[:, :self.t_max*(self.index+2)].sum(1) == 0 pass
        
        return spikes
