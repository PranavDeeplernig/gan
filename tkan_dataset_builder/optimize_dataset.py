import torch
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_sharded_pt(dir_path):
    """Loads and merges sharded .pt files from a directory."""
    logging.info(f"Loading shards from {dir_path}...")
    shards = sorted([f for f in os.listdir(dir_path) if f.endswith('.pt')])
    all_data = []
    for shard in shards:
        path = os.path.join(dir_path, shard)
        all_data.extend(torch.load(path, weights_only=False))
    return all_data

def convert_to_tensor_format(input_path, output_path, max_nodes=40):
    if os.path.isdir(input_path):
        raw_data = load_sharded_pt(input_path)
    else:
        print(f"Loading {input_path}... (This might take RAM, but only once)")
        raw_data = torch.load(input_path, weights_only=False)
    
    num_seq = len(raw_data)
    seq_len = 20 # As defined in your config
    node_dim = 4
    agent_dim = 3
    
    # Pre-allocate HUGE tensors (Compact Memory)
    # [N_Seqs, T, Max_Nodes, F]
    X_tensor = torch.zeros((num_seq, seq_len, max_nodes, node_dim), dtype=torch.float32)
    Agent_tensor = torch.zeros((num_seq, seq_len, max_nodes, agent_dim), dtype=torch.float32)
    Y_tensor = torch.zeros((num_seq,), dtype=torch.long)
    
    # Adjacency matrices for speed
    # [N_Seqs, T, Max_Nodes, Max_Nodes] - 1 if connected, 0 if not
    Adj_tensor = torch.zeros((num_seq, seq_len, max_nodes, max_nodes), dtype=torch.bool)

    print("Converting to Dense Tensors...")
    for i, seq_item in enumerate(tqdm(raw_data)):
        seq_data = seq_item['x'] # List of Data objects
        y_label = seq_item['y']
        
        # y can be a tensor or a number
        if isinstance(y_label, torch.Tensor):
            Y_tensor[i] = y_label.item()
        else:
            Y_tensor[i] = y_label
        
        for t, snapshot in enumerate(seq_data):
            num_nodes = snapshot.x.shape[0]
            limit = min(num_nodes, max_nodes)
            
            # 1. Copy Node Features
            X_tensor[i, t, :limit, :] = snapshot.x[:limit]
            
            # 2. Copy Agent Features
            Agent_tensor[i, t, :limit, :] = snapshot.global_p[:limit]
            
            # 3. Fill Adjacency
            edges = snapshot.edge_index
            mask = (edges[0] < limit) & (edges[1] < limit)
            valid_edges = edges[:, mask]
            Adj_tensor[i, t, valid_edges[0], valid_edges[1]] = 1

    print(f"Saving optimized dataset to {output_path}...")
    torch.save({
        'x': X_tensor,
        'adj': Adj_tensor,
        'agent': Agent_tensor,
        'y': Y_tensor
    }, output_path)
    print("Done! Optimization complete.")

if __name__ == "__main__":
    train_in = 'graph_data_output/train_graphs'
    train_out = 'graph_data_output/train_tensor.pt'
    val_in = 'graph_data_output/val_graphs'
    val_out = 'graph_data_output/val_tensor.pt'

    if os.path.isdir(train_in) or os.path.exists(train_in + ".pt"):
        inp = train_in if os.path.isdir(train_in) else train_in + ".pt"
        convert_to_tensor_format(inp, train_out)
    else:
        print(f"Missing {train_in}")
        
    if os.path.isdir(val_in) or os.path.exists(val_in + ".pt"):
        inp = val_in if os.path.isdir(val_in) else val_in + ".pt"
        convert_to_tensor_format(inp, val_out)
    else:
        print(f"Missing {val_in}")
