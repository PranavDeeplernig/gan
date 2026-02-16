import torch
import os
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def chunk_file(file_path, chunk_size_mb=45):
    """
    Splits a large .pt file into smaller chunks.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return

    logging.info(f"Chunking {file_path}...")
    data = torch.load(file_path, weights_only=False)
    
    # We assume 'data' is a list (like the original sparse graphs)
    # If it's the optimized dict, we handle it differently, but here 
    # we aim to chunk the source graphs which are the ones causing Git issues.
    
    if isinstance(data, list):
        num_items = len(data)
        # Estimate size per item
        total_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        num_chunks = math.ceil(total_size_mb / chunk_size_mb)
        items_per_chunk = math.ceil(num_items / num_chunks)
        
        base_name = os.path.splitext(file_path)[0]
        os.makedirs(base_name, exist_ok=True)
        
        for i in range(num_chunks):
            start = i * items_per_chunk
            end = min((i + 1) * items_per_chunk, num_items)
            chunk = data[start:end]
            
            chunk_path = os.path.join(base_name, f"part_{i}.pt")
            torch.save(chunk, chunk_path)
            logging.info(f"Saved {chunk_path} ({len(chunk)} items)")
            
        logging.info(f"Successfully split into {num_chunks} chunks in {base_name}/")
        
        # We don't delete original here, let the user handle it or the main script
    else:
        logging.error("Unsupported data format for chunking (expected list of graphs).")

if __name__ == "__main__":
    import sys
    # Handle the two main files
    chunk_file('graph_data_output/train_graphs.pt')
    chunk_file('graph_data_output/val_graphs.pt')
