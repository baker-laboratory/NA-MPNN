# The code in this file was adapted/borrowed from:
# https://www.kaggle.com/code/shujun717/ribonanzanet-2d-structure-inference
# https://www.kaggle.com/code/shujun717/ribonanzanet-inference
import contextlib
import os
import sys
import tempfile
import yaml

import numpy as np
import torch

sys.path.append("/projects/ml/afavor/ribonanzanet/kaggle/input/RibonanzaNet2D_Final")
from Network import RibonanzaNet

# Setup ARNIE.
sys.path.append("/projects/ml/afavor/ribonanzanet/")
with tempfile.NamedTemporaryFile(mode = "wt", suffix = ".txt") as f:
    # Setup the ARNIE config file.
    f.write("linearpartition: . \nTMP: /tmp")
    f.flush()
    arnie_config_path = f.name
    os.environ["ARNIEFILE"] = arnie_config_path

    # Import the _hungarian method from ARNIE.
    from arnie.pk_predictors import _hungarian

# Constants.
rna_restype_to_int = {
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3
}

# Device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ribonanza_net_seq_to_tensor(sequence):
    """
    Given a sequence, converts the sequence to a tensor for input into the
    RibonanzaNet algorithm.

    Args:
        sequence (str): The sequence to convert to a tensor.
    
    Returns:
        sequence_tensor (torch.Tensor): The sequence as a tensor.
    """
    # Check that the RNA sequence is valid.
    for c in sequence:
        if c not in rna_restype_to_int:
            raise ValueError(f"Invalid RNA sequence: {sequence}")
        
    # Convert the sequence to a tensor.
    sequence_tensor = torch.tensor([rna_restype_to_int[restype_1] for restype_1 in sequence]).unsqueeze(0)

    return sequence_tensor

# Config for RibonanzaNet.
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

# Loading function for RibonanzaNet config.
def load_config_from_yaml(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return Config(**config)

# Class for 2D RibonanzaNet.
class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config):
        config.dropout = 0.3
        super(finetuned_RibonanzaNet, self).__init__(config)

        self.dropout = torch.nn.Dropout(0.0)
        self.ct_predictor = torch.nn.Linear(64, 1)

    def forward(self,src):
        
        _, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))

        pairwise_features=pairwise_features+pairwise_features.permute(0, 2, 1, 3) #symmetrize

        output=self.ct_predictor(self.dropout(pairwise_features)) #predict

        return output.squeeze(-1)

def run_ribonanza_net_reactivity_profile(sequence,
                                         model,
                                         batch_size = 1):
    """
    Given a sequence, runs the RibonanzaNet algorithm to predict the reactivity
    profile of the sequence.

    Args:
        sequence (str): The sequence to predict the reactivity profile for.
        model (torch.nn.Module): The RibonanzaNet model to use for prediction.
        batch_size (int): The number of samples to predict in a batch.
    
    Returns:
        result (dict): A dictionary containing:
            predicted_2A3_reactivity_profiles (list of float lists): A list of
                predicted reactivity profiles of the sequence for the 2A3 probe.
            predicted_DMS_reactivity_profiles (list of float lists): A list of
                predicted reactivity profiles of the sequence for the DMS probe.
    """    
    # Convert the sequence to a tensor.
    seq_tensor = ribonanza_net_seq_to_tensor(sequence)

    # Predict the reactivity profile.
    predicted_2A3_reactivity_profiles = []
    predicted_DMS_reactivity_profiles = []
    for i in range(batch_size):
        predicted_reactivity_profile = \
            model(seq_tensor, torch.ones_like(seq_tensor).detach())
        predicted_2A3_reactivity_profiles.append(predicted_reactivity_profile.detach().numpy()[0, :, 0].tolist())
        predicted_DMS_reactivity_profiles.append(predicted_reactivity_profile.detach().numpy()[0, :, 1].tolist())
    
    result = {
        "predicted_2A3_reactivity_profiles": predicted_2A3_reactivity_profiles,
        "predicted_DMS_reactivity_profiles": predicted_DMS_reactivity_profiles
    }

    return result

def run_ribonanza_net_secondary_structure(sequence,
                                          model,
                                          batch_size = 1):
    """
    Given a sequence, runs the RibonanzaNet algorithm to predict the secondary
    structure of the sequence.

    Args:
        sequence (str): The sequence to predict the secondary structure for.
        model (torch.nn.Module): The RibonanzaNet model to use for prediction.
        batch_size (int): The number of samples to predict in a batch.
    
    Returns:
        result (dict): A dictionary containing:
            predicted_secondary_structures (str list): The predicted secondary
                structures of the sequence.
            predicted_base_pairs (list of int tuples): The predicted base pairs
                of the sequence.
            predicted_base_pair_matrix (numpy.ndarray): The predicted base pair
                matrix of the sequence.
    """    
    # Function for masking the near-diagonal elements of a matrix.
    def mask_diagonal(matrix, mask_value = 0):
        matrix = matrix.copy()
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if abs(i - j) < 4:
                    matrix[i][j] = mask_value
        return matrix
    
    # Convert the sequence to a tensor.
    seq_tensor = ribonanza_net_seq_to_tensor(sequence)
    
    # Predict the secondary structure.
    hungarian_matrix_list = []
    hungarian_secondary_structure_list = []
    hungarian_base_pair_list = []
    for i in range(batch_size):
        # Predict the probabilities of pairing.
        predicted_logits = model(seq_tensor).sigmoid().cpu().detach().numpy()[0]

        # Run the Hungarian algorithm to determine the secondary structure.
        hungarian_secondary_structure, hungarian_base_pairs = \
            _hungarian(mask_diagonal(predicted_logits), 
                       theta = 0.5, 
                       min_len_helix = 1)

        # Compute the base pair matrix.
        hungarian_matrix = np.zeros((len(hungarian_secondary_structure), 
                                     len(hungarian_secondary_structure)))
        for bp in hungarian_base_pairs:
            hungarian_matrix[bp[0], bp[1]] = 1
        hungarian_matrix = hungarian_matrix + hungarian_matrix.T

        # Append the results to the lists.
        hungarian_matrix_list.append(hungarian_matrix)
        hungarian_secondary_structure_list.append(hungarian_secondary_structure)
        hungarian_base_pair_list.append(hungarian_base_pairs)

    result = {
        "predicted_secondary_structures": hungarian_secondary_structure_list,
        "predicted_base_pairs": hungarian_base_pair_list,
        "predicted_base_pair_matrix": hungarian_matrix_list
    }

    return result

if __name__ == "__main__":
    mode = sys.argv[1]
    sequence = sys.argv[2]
    output_directory = sys.argv[3]
    batch_size = int(sys.argv[4])

    config_path = "/projects/ml/afavor/ribonanzanet/kaggle/input/RibonanzaNet2D_Final/configs/pairwise.yaml"
    if mode == "reactivity_profile":
        model_class = RibonanzaNet
        weights_path = "/projects/ml/afavor/ribonanzanet/kaggle/input/RibonanzaNet_Weights/RibonanzaNet.pt"
    elif mode == "secondary_structure":
        model_class = finetuned_RibonanzaNet
        weights_path = "/projects/ml/afavor/ribonanzanet/kaggle/input/RibonanzaNet_Weights/RibonanzaNet-SS.pt"
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Load the RibonanzaNet model.
    model = model_class(load_config_from_yaml(config_path))
    model.load_state_dict(torch.load(weights_path, map_location = device))
    model.eval()

    if mode == "reactivity_profile":
        result = run_ribonanza_net_reactivity_profile(sequence,
                                                      model,
                                                      batch_size = batch_size)
    elif mode == "secondary_structure":
        result = run_ribonanza_net_secondary_structure(sequence,
                                                       model,
                                                       batch_size = batch_size)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Save the result.
    output_path = os.path.join(output_directory, "output.npy")
    np.save(output_path, result)