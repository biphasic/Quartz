import sinabs.layers as sl
import torch
import torch.nn as nn


class IF(sl.StatefulLayer):
    def __init__(
        self,
        t_max: int = 128,
    ):
        super().__init__(state_names=["v_mem", "i_syn"])
        self.t_max = t_max

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, *trailing_dim = data.shape

        # Ensure the neuron state are initialized
        if not self.is_state_initialised() or not self.state_has_shape(
            (batch_size, *trailing_dim)
        ):
            self.init_state_with_shape((batch_size, *trailing_dim))

        self.i_syn = self.i_syn + data

        self.v_mem = self.v_mem + self.i_syn

        # spikes = torch.zeros_like(data)
        spikes = (self.v_mem >= self.t_max).float()

        return spikes