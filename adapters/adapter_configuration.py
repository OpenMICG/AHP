# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Implements the adapter's configuration."""
from dataclasses import dataclass 

@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""
    # This is for the layernorms applied after feedforward/self-attention layers.
    add_layer_norm_before_adapter: bool = True
    add_layer_norm_after_adapter: bool = True
    nonlinearity: str = "gelu_new"
    reduction_factor: int = 16
    # By default, we add adapters after attention, set False if otherwise.
    add_adapter_after_attention = True
    add_adapter_after_feedforward = True
    # Trains the adapters if this is set to true.
    adapter_tune = False   
