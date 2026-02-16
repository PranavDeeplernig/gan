import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KANLinear(nn.Module):
    """
    Efficient KAN Linear layer using B-Splines.
    Learns Univariate activation functions on edges: y = splines(x) + base_weight(x).
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Base weight (linear part)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))

        # Spline coefficients
        # Shape: [out_features, in_features, grid_size + spline_order]
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        nn.init.normal_(self.spline_weight, mean=0.0, std=scale_noise / np.sqrt(in_features))

        # Grid points
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        # Extend grid for splines
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.cat([
            torch.linspace(grid_range[0] - spline_order * h, grid_range[0] - h, spline_order),
            grid,
            torch.linspace(grid_range[1] + h, grid_range[1] + spline_order * h, spline_order)
        ])
        self.register_buffer("grid", grid)

        self.scale_base = scale_base
        self.scale_spline = scale_spline

    def b_splines(self, x):
        """
        Compute B-spline basis functions for x.
        x: [B, ..., in_features]
        """
        grid = self.grid # [grid_size + 2*spline_order + 1]
        x = x.unsqueeze(-1) # [B, ..., in_features, 1]
        
        # 0-order basis
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()

        # Higher-order recursion
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[: -(k + 1)]) / (grid[k:-1] - grid[: -(k + 1)]) * bases[..., :-1]
                + (grid[k + 1 :] - x) / (grid[k + 1 :] - grid[1:-k]) * bases[..., 1:]
            )
        return bases # [B, ..., in_features, grid_size + spline_order]

    def forward(self, x):
        # Handle N-D tensors by flattening all but the last dimension
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features) # [Flat, In]

        # Base linear activation
        base_output = F.linear(F.silu(x), self.base_weight) * self.scale_base

        # Spline activation
        splines = self.b_splines(x) # [Flat, In, Basis]
        spline_output = torch.einsum("bin,oin->bo", splines, self.spline_weight) * self.scale_spline

        out = base_output + spline_output
        
        # Reshape back to [B, ..., Out]
        new_shape = list(orig_shape[:-1]) + [self.out_features]
        return out.reshape(*new_shape)

    @torch.no_grad()
    def plot_splines(self):
        """
        Visualize the univariate activation functions for each edge.
        """
        import matplotlib.pyplot as plt
        x_range = torch.linspace(self.grid[0], self.grid[-1], 100).to(self.grid.device)
        # Compute basis for x_range: [100, 1] -> [100, 1, n_basis]
        # We treat the range as a single 'pseudo-input' for calculation
        bases = self.b_splines(x_range.unsqueeze(-1)) 
        
        # Activations = sum(bases * spline_weight)
        # bases: [100, 1, n_basis]
        # weight: [out, in, n_basis]
        # result: [100, out, in]
        # We want to multiply the weights for each input by the same spline basis
        activations = torch.einsum("btn,oin->boi", bases, self.spline_weight)
        activations = activations.cpu().numpy()
        x_np = x_range.cpu().numpy()
        
        fig, axes = plt.subplots(self.out_features, self.in_features, figsize=(self.in_features*2, self.out_features*2))
        if self.out_features == 1 and self.in_features == 1:
            axes = np.array([[axes]])
        elif self.out_features == 1 or self.in_features == 1:
            axes = axes.reshape(self.out_features, self.in_features)
            
        for i in range(self.out_features):
            for j in range(self.in_features):
                ax = axes[i, j]
                ax.plot(x_np, activations[:, i, j], color='#E63946')
                ax.grid(True, alpha=0.2)
                if i == 0: ax.set_title(f"In {j}")
                if j == 0: ax.set_ylabel(f"Out {i}")
        
        plt.tight_layout()
        return fig

class GatedTKAN(nn.Module):
    """
    Temporal-KAN with GLU Gating.
    iv_velocity (Stress) acts as the 'gate' that controls GEX signal attention.
    """
    def __init__(self, n_feat, lstm_hidden=64, kan_hidden=[16, 8], n_classes=3, grid_size=5, spline_order=3, dropout=0.5):
        super(GatedTKAN, self).__init__()
        
        self.lstm = nn.LSTM(n_feat, lstm_hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(lstm_hidden)
        
        # Explicit GLU Gating based on iv_velocity
        self.iv_gate = nn.Sequential(
            nn.Linear(1, 8),
            nn.SiLU(),
            nn.Linear(8, lstm_hidden),
            nn.Sigmoid()
        )
        # Initialize gate bias to 0 to start with 0.5 gate (Neutral)
        nn.init.constant_(self.iv_gate[2].bias, 0.0)
        
        # Multi-layer KAN Sequential (Bottleneck)
        layers = []
        curr_in = lstm_hidden
        
        if isinstance(kan_hidden, (list, tuple)):
            for h in kan_hidden:
                layers.append(KANLinear(curr_in, h, grid_size=grid_size, spline_order=spline_order))
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
                curr_in = h
        else:
            layers.append(KANLinear(curr_in, kan_hidden, grid_size=grid_size, spline_order=spline_order))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            curr_in = kan_hidden
            
        layers.append(KANLinear(curr_in, n_classes, grid_size=grid_size, spline_order=spline_order))
        self.kan = nn.Sequential(*layers)
        
    def forward(self, x, iv_velocity=None):
        """
        x: [batch, seq_len, n_feat]
        iv_velocity: [batch] (last known dIV)
        """
        # LSTM Encoding
        out, _ = self.lstm(x)
        h = out[:, -1, :] # Last hidden state [batch, lstm_hidden]
        h = self.ln(h)    # Stabilize
        
        # Disable Gating for "Pure T-KAN" feature verification
        # gate = self.iv_gate(iv_velocity.unsqueeze(-1))
        # h = h * gate
        
        # KAN Sequential Pass
        return self.kan(h)

    def regularization_loss(self):
        reg = 0
        for m in self.kan:
            if hasattr(m, 'regularization_loss'):
                reg += m.regularization_loss()
        return reg

class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    L = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # [C]
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply class weights (alpha)
        # targets: [B]
        at = self.alpha.gather(0, targets)
        
        focal_loss = at * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

class VolatilityWeightedCrossEntropy(nn.Module):
    """
    L = CE * (1 + k * |dIV|)
    Penalizes errors during high stress (high dIV) more heavily.
    """
    def __init__(self, weight=None, k=5.0):
        super(VolatilityWeightedCrossEntropy, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.k = k
        
    def forward(self, logits, targets, iv_velocity):
        loss = self.ce(logits, targets)
        weight = 1.0 + self.k * torch.abs(iv_velocity)
        return (loss * weight).mean()
