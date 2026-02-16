import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax

# Reuse KANLinear from the core implementation
import sys
import os
sys.path.append(os.getcwd())
from core.tkan_model import KANLinear

class KANGATConv(nn.Module):
    """
    Graph Attention Network layer using Force Law logic (Dense Version).
    Energy_ij = KAN_force(h_i - h_j)
    """
    def __init__(self, in_channels, out_channels, grid_size=5, spline_order=3):
        super(KANGATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # KAN for Attention mechanism (Force Law)
        self.kan_force = KANLinear(in_channels, 1, grid_size=grid_size, spline_order=spline_order)
        # KAN for Message transformation
        self.kan_msg = KANLinear(in_channels, out_channels, grid_size=grid_size, spline_order=spline_order)
        # KAN for final node update
        self.kan_update = KANLinear(in_channels + out_channels, out_channels, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x, adj):
        # x: [B, N, F], adj: [B, N, N]
        B, N, F = x.shape
        
        # 1. Pairwise differences for Force field: [B, N, N, F]
        x_i = x.unsqueeze(2) # [B, N, 1, F]
        x_j = x.unsqueeze(1) # [B, 1, N, F]
        r_ij = x_i - x_j     # [B, N, N, F]
        
        # 2. Energy (Force) calculation
        energy = self.kan_force(r_ij).squeeze(-1) # [B, N, N]
        
        # 3. Mask out non-edges
        energy = energy.masked_fill(adj == 0, -1e9)
        alpha = torch.softmax(energy, dim=-1) # [B, N, N]
        
        # 4. Message Passing
        msg_val = self.kan_msg(x) # [B, N, Out]
        aggr_out = torch.bmm(alpha, msg_val) # [B, N, Out]
        
        # 5. Update
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.kan_update(combined)

class KANGRUCell(nn.Module):
    """
    Gated Recurrent Unit cell where linear transforms are replaced by KAN.
    """
    def __init__(self, input_size, hidden_size, grid_size=5, spline_order=3):
        super(KANGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # z_t = sigmoid(Phi_z(h_t-1 + x_t))
        self.kan_z = KANLinear(input_size + hidden_size, hidden_size, grid_size=grid_size, spline_order=spline_order)
        # r_t = sigmoid(Phi_r(h_t-1 + x_t))
        self.kan_r = KANLinear(input_size + hidden_size, hidden_size, grid_size=grid_size, spline_order=spline_order)
        # h_tilde = tanh(Phi_h(x_t + r_t * h_t-1))
        self.kan_h = KANLinear(input_size + hidden_size, hidden_size, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x, h_prev):
        # x: [B, input_size], h_prev: [B, hidden_size]
        combined = torch.cat([x, h_prev], dim=-1)
        
        z = torch.sigmoid(self.kan_z(combined))
        r = torch.sigmoid(self.kan_r(combined))
        
        # New memory candidate
        combined_h = torch.cat([x, r * h_prev], dim=-1)
        h_tilde = torch.tanh(self.kan_h(combined_h))
        
        # Final state
        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next

class AlphaGammaNet(nn.Module):
    """
    AG-SGN: Alpha Gamma Spatiotemporal Graph Network (Dense Version).
    """
    def __init__(self, node_in=4, global_in=3, hidden_dim=32, n_classes=3, 
                 grid_size=5, spline_order=3, n_heads=4):
        super(AlphaGammaNet, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Step 1: KAN-Encoder (Node Lens)
        self.node_encoder = KANLinear(node_in, hidden_dim, grid_size=grid_size, spline_order=spline_order)
        
        # Step 2: KAN-GAT Interaction
        self.gnn = KANGATConv(hidden_dim, hidden_dim, grid_size=grid_size, spline_order=spline_order)
        
        # Step 3: T-KAN (Temporal Memory) - Global State
        self.tkan_cell = KANGRUCell(hidden_dim, hidden_dim, grid_size=grid_size, spline_order=spline_order)
        
        # Step 4: Price Agent Readout
        self.agent_encoder = nn.Linear(global_in, hidden_dim)
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)
        
        self.class_head = nn.Sequential(
            KANLinear(hidden_dim, 16, grid_size=grid_size, spline_order=spline_order),
            nn.SiLU(),
            KANLinear(16, n_classes, grid_size=grid_size, spline_order=spline_order)
        )
        
        self.aux_vol_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_seq, adj_seq, agent_seq, h_temporal=None):
        """
        Inputs:
        - x_seq: [B, T, N, F]
        - adj_seq: [B, T, N, N]
        - agent_seq: [B, T, N, G]
        - h_temporal: Optional initial state [B, Hidden]
        """
        B, T, N, _ = x_seq.shape
        
        # Default Temporal State
        if h_temporal is None:
            h_temporal = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        elif h_temporal.shape[0] != B:
            h_temporal = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        
        all_snapshots_nodes = [] 
        
        for t in range(T):
            xt = x_seq[:, t, :, :] # [B, N, F]
            adjt = adj_seq[:, t, :, :] # [B, N, N]
            
            # Encode and GNN
            ht = self.node_encoder(xt) # [B, N, Hidden]
            ht = self.gnn(ht, adjt)     # [B, N, Hidden]
            
            node_states = ht
            all_snapshots_nodes.append(node_states)
            
            # T-KAN Step
            graph_vec = node_states.mean(dim=1) 
            h_temporal = self.tkan_cell(graph_vec, h_temporal)
            
        graph_memory = torch.stack(all_snapshots_nodes, dim=1)
        
        target_agent = agent_seq[:, -1, :, :] 
        aq = self.agent_encoder(target_agent) 
        kv = graph_memory[:, -1, :, :]        
        
        attn_out, _ = self.mha(aq, kv, kv)
        net_force = attn_out.mean(dim=1) 
        
        final_embedding = net_force + h_temporal
        
        return self.class_head(final_embedding), self.aux_vol_head(final_embedding), h_temporal
