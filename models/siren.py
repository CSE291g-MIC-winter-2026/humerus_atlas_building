"""
siren.py — SIREN (Sinusoidal Representation Network) for bone SDF.

Adapted from CINeMA's SIREN implementation. Uses periodic activations
(sin) with FiLM-style latent modulation for subject-specific decoding.

Reference: Sitzmann et al., "Implicit Neural Representations with
           Periodic Activation Functions", NeurIPS 2020
"""

import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """
    Single SIREN layer with optional FiLM-style latent modulation.
    
    Computes: sin(ω * (W·x + b) * γ + β)
    where γ, β are predicted from the latent code (if provided).
    
    Args:
        in_feat: Input feature dimension
        lat_feat: Latent modulation dimension (0 = no modulation)
        out_feat: Output feature dimension
        bias: Whether to use bias in linear layers
        is_first: If True, uses different weight initialization
        omega: Frequency scaling factor for sin activation
    """
    
    def __init__(self, in_feat, lat_feat, out_feat, bias=True,
                 is_first=False, omega=30):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.in_features = in_feat
        self.out_features = out_feat
        
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        self.linear_lats = (
            nn.Linear(lat_feat, out_feat * 2, bias=bias)
            if lat_feat > 0 else None
        )
        self.init_weights()
    
    def init_weights(self):
        """SIREN-specific weight initialization."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features,
                    1 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega,
                    np.sqrt(6 / self.in_features) / self.omega
                )
    
    def forward(self, input):
        """
        Args:
            input: Tuple of (coords, latent_vec)
                coords: (N, in_feat)
                latent_vec: (N, lat_feat)
        
        Returns:
            Tuple of (output, latent_vec)
        """
        intermed = self.linear(input[0])
        
        if self.linear_lats is not None:
            lats = self.linear_lats(input[1])
            # FiLM modulation: gamma * x + beta
            gamma = lats[..., :self.out_features]
            beta = lats[..., self.out_features:]
            out = torch.sin((self.omega * intermed * gamma) + beta)
        else:
            out = torch.sin(self.omega * intermed)
        
        return out, input[1]


class Siren(nn.Module):
    """
    Full SIREN network with FiLM-style latent modulation.
    
    Architecture:
        SineLayer_0(in→hidden) → SineLayer_1(hidden→hidden) → ... → Linear(hidden→out)
    
    Selected layers can be modulated by a per-subject latent code,
    enabling the auto-decoder to represent subject-specific anatomy.
    
    Args:
        in_size: Input coordinate dimension (3 for 3D)
        lat_size: Latent code channel dimension
        out_size: Output dimension (1 for SDF)
        hidden_size: Width of hidden layers
        num_layers: Number of hidden SIREN layers
        f_om: First layer omega frequency
        h_om: Hidden layer omega frequency
        outermost_linear: If True, final layer is linear (no sin)
        modulated_layers: List of layer indices to receive latent modulation
    """
    
    def __init__(self, in_size, lat_size, out_size, hidden_size,
                 num_layers, f_om, h_om, outermost_linear,
                 modulated_layers):
        super().__init__()
        
        # First layer: coords → hidden
        l_in_mod = 0 in modulated_layers
        self.net = [
            SineLayer(in_size, lat_size * l_in_mod, hidden_size,
                      is_first=True, omega=f_om)
        ]
        self.hidden_size = hidden_size
        
        # Hidden layers
        for i in range(num_layers):
            l_in_mod = (i + 1) in modulated_layers
            self.net.append(
                SineLayer(hidden_size, lat_size * l_in_mod, hidden_size,
                          is_first=False, omega=h_om)
            )
        
        # Final layer
        if outermost_linear:
            self.final_linear = nn.Linear(hidden_size + lat_size, out_size, bias=True)
            with torch.no_grad():
                self.final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_size) / h_om,
                    np.sqrt(6 / hidden_size) / h_om
                )
        else:
            self.final_linear = SineLayer(
                hidden_size, 0, out_size, is_first=False, omega=h_om
            )
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        """
        Args:
            x: Tuple of (coords, latent_modulations)
        
        Returns:
            Output tensor of shape (N, out_size)
        """
        x = self.net(x)
        return self.final_linear(torch.cat([x[0], x[1]], dim=-1))
