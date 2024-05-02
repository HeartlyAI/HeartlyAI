import torch
import argparse

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Extract the state dictionary from a PyTorch checkpoint and save it.')
parser.add_argument('ckpt_path', type=str, help='Path to the model checkpoint file (e.g., model.ckpt)')
parser.add_argument('output_path', type=str, help='Path where the state dictionary should be saved (e.g., state_dict.pth)')
args = parser.parse_args()

# Load the checkpoint
checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cpu'))

# Extract the state_dict
# The state_dict might be stored directly or under a key, depending on how the checkpoint was created
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# Save the state_dict directly
torch.save(state_dict, args.output_path)
