import os
import torch
from collections import OrderedDict


def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0


def load_pretrained_encoder(encoder, pretrained_file, verbose=True):
    if os.path.exists(pretrained_file):
        pretrained_state_dict = torch.load(pretrained_file)
    else:
        raise FileNotFoundError(f"Pretrained file not found: {pretrained_file}")

    # Create a new state dict
    new_state_dict = OrderedDict()

    # Get the current encoder state dict
    encoder_state_dict = encoder.state_dict()

    # Iterate through the pre-trained state dict
    for k, v in pretrained_state_dict.items():
        # Check if the key exists in the encoder state dict
        if k in encoder_state_dict:
            new_state_dict[k] = v
        else:
            # Try to find a matching key
            found = False
            for encoder_key in encoder_state_dict.keys():
                if k.endswith(encoder_key):
                    new_state_dict[encoder_key] = v
                    found = True
                    if verbose:
                        print(f"Matched {k} to {encoder_key}")
                    break
            if not found and verbose:
                print(f"Could not find a match for {k}")

    # Load the new state dict into the encoder
    encoder.load_state_dict(new_state_dict, strict=False)

    if verbose:
        print("Pretrained weights loaded successfully")

    return encoder
