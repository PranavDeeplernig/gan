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

def convert_to_tensor_format(input_path, max_nodes=40):
    if os.path.isdir(input_path):
        raw_data = load_sharded_pt(input_path)
    else:
        print(f"Loading {input_path}... (This might take RAM, but only once)")
        raw_data = torch.load(input_path, weights_only=False)
    
    num_seq = len(raw_data)
    seq_len = 20
    node_dim = 4
    agent_dim = 3
    
    X_tensor = torch.zeros((num_seq, seq_len, max_nodes, node_dim), dtype=torch.float32)
    Agent_tensor = torch.zeros((num_seq, seq_len, max_nodes, agent_dim), dtype=torch.float32)
    Y_tensor = torch.zeros((num_seq,), dtype=torch.long)
    Adj_tensor = torch.zeros((num_seq, seq_len, max_nodes, max_nodes), dtype=torch.bool)

    print(f"Processing {num_seq} sequences...")
    for i, seq_item in enumerate(tqdm(raw_data)):
        seq_data = seq_item['x']
        y_label = seq_item['y']
        
        if isinstance(y_label, torch.Tensor):
            Y_tensor[i] = y_label.item()
        else:
            Y_tensor[i] = y_label
        
        for t, snapshot in enumerate(seq_data):
            num_nodes = snapshot.x.shape[0]
            limit = min(num_nodes, max_nodes)
            X_tensor[i, t, :limit, :] = snapshot.x[:limit]
            Agent_tensor[i, t, :limit, :] = snapshot.global_p[:limit]
            
            edges = snapshot.edge_index
            mask = (edges[0] < limit) & (edges[1] < limit)
            valid_edges = edges[:, mask]
            Adj_tensor[i, t, valid_edges[0], valid_edges[1]] = 1

    return {
        'x': X_tensor,
        'adj': Adj_tensor,
        'agent': Agent_tensor,
        'y': Y_tensor
    }

def normalize_and_save(train_data, val_data, train_out, val_out, stats_out):
    logging.info("Applying Rolling Z-Score Normalization (Past 20 Points)...")
    
    def apply_rolling_norm(tensor):
        """
        tensor: [B, T, N, F]
        Normalizes each window [T, N] for each feature F independently.
        """
        # Mean across T (time) and N (nodes) to get local regime stats
        # result: [B, 1, 1, F]
        mu = tensor.mean(dim=(1, 2), keepdim=True)
        std = tensor.std(dim=(1, 2), keepdim=True) + 1e-9
        
        return (tensor - mu) / std

    # Apply rolling normalization to Node features
    logging.info("Normalizing Node Features...")
    train_data['x'] = apply_rolling_norm(train_data['x'])
    val_data['x'] = apply_rolling_norm(val_data['x'])
    
    # Apply rolling normalization to Agent features
    logging.info("Normalizing Agent Features...")
    train_data['agent'] = apply_rolling_norm(train_data['agent'])
    val_data['agent'] = apply_rolling_norm(val_data['agent'])
    
    # Still save global stats as a reference/sanity check
    x_mean_global = train_data['x'].mean()
    logging.info(f"Global mean after rolling norm: {x_mean_global.item():.6f}")
    
    logging.info(f"Saving rolling-normalized datasets to {train_out} and {val_out}...")
    torch.save(train_data, train_out)
    torch.save(val_data, val_out)
    
    # norm_stats.pth now stores global reference but is optional for inference 
    # as inference will normalize its own window
    torch.save({'type': 'rolling_local', 'window_size': 20}, stats_out)

if __name__ == "__main__":
    train_in = 'graph_data_output/train_graphs'
    val_in = 'graph_data_output/val_graphs'
    
    train_out = 'graph_data_output/train_tensor.pt'
    val_out = 'graph_data_output/val_tensor.pt'
    stats_out = 'graph_data_output/norm_stats.pth'

    if (os.path.isdir(train_in) or os.path.exists(train_in + ".pt")) and \
       (os.path.isdir(val_in) or os.path.exists(val_in + ".pt")):
        
        t_inp = train_in if os.path.isdir(train_in) else train_in + ".pt"
        v_inp = val_in if os.path.isdir(val_in) else val_in + ".pt"
        
        train_data = convert_to_tensor_format(t_inp)
        val_data = convert_to_tensor_format(v_inp)
        
        normalize_and_save(train_data, val_data, train_out, val_out, stats_out)
        print("Done! Normalization and Optimization complete.")
    else:
        print("Missing required graph directories.")
