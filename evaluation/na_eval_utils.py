################################################################################
# Imports
################################################################################
# Python Standard Libraries
import argparse
import ast
import copy
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

# Third-Party Libraries
import numpy as np
import pandas as pd

################################################################################
# Common Functions
################################################################################
def read_text_file(path):
    """
    Given a path to a text file, reads the file and returns the contents as a
    string.

    Args:
        path (str): The path to the text file to read.
    
    Returns:
        contents (str): The contents of the file as a string.
    """
    with open(path, mode = "rt") as f:
        contents = f.read()
        return contents

def write_text_file(path, contents):
    """
    Given a path and contents, writes the contents to the file at the given 
    path.

    Args:
        path (str): The path to the file to write.
        contents (str): The contents to write to the file.
    
    Side Effects:
        Writes the contents to the file at the given path.
    """
    with open(path, mode = "wt") as f:
        f.write(contents)

def read_cluster_ids_text_file(path):
    """
    Read a text file containing cluster IDs and return a list of the cluster
    IDs as integers.

    Args:
        path (str): The path to the text file containing cluster IDs.
    
    Returns:
        cluster_ids (int list): A list of the cluster IDs as integers.
    """
    cluster_ids_text = read_text_file(path)
    cluster_ids = cluster_ids_text.strip().split("\n")
    cluster_ids = [int(cluster_id) for cluster_id in cluster_ids]
    return cluster_ids

def read_json_file(path):
    """
    Given a path to a json file, reads the file and returns the contents as a
    dictionary.

    Args:
        path (str): The path to the json file to read.
    
    Returns:
        contents (dict): The contents of the file as a dictionary.
    """
    with open(path, mode = "rt") as f:
        contents = json.load(f)
        return contents

def write_json_file(path, contents):
    """
    Given a path and contents, writes the contents to the file at the given 
    path.

    Args:
        path (str): The path to the file to write.
        contents (dict): The contents to write to the file.
    
    Side Effects:
        Writes the contents to the file at the given path.
    """
    with open(path, mode = "wt") as f:
        json.dump(contents, f, indent = 4)

def read_fasta_file(path):
    """
    Given a path to a fasta file, reads the file and returns a list of tuples,
    where each tuple contains the header and sequence of a fasta entry.

    Args:
        path (str): The path to the fasta file to read.

    Returns:
        fasta_entries ((str, str) list): A list of tuples, where each tuple
            contains the header and sequence of a fasta entry.
    """
    fasta_text = read_text_file(path)

    fasta_text = fasta_text.strip()
    
    if fasta_text.startswith(">"):
        fasta_text = fasta_text[1:]

    fasta_lines = fasta_text.split("\n>")

    fasta_entries = []
    for fasta_line in fasta_lines:
        fasta_line = fasta_line.strip()
        
        fasta_header, fasta_sequence = fasta_line.split("\n", 1)

        fasta_header = fasta_header.strip()
        fasta_sequence = fasta_sequence.strip()

        fasta_entries.append((fasta_header, fasta_sequence))
    
    return fasta_entries

def write_fasta_file(path, fasta_entries):
    """
    Given a path and a list of tuples, where each tuple contains the header and
    sequence of a fasta entry, writes the fasta entries to the file at the given
    path.

    Args:
        path (str): The path to the fasta file to write.
        fasta_entries ((str, str) list): A list of tuples, where each tuple
            contains the header and sequence of a fasta entry.
    
    Side Effects:
        Writes the fasta entries to the file at the given path.
    """
    fasta_lines = []
    for fasta_header, fasta_sequence in fasta_entries:
        fasta_line = f">{fasta_header}\n{fasta_sequence}"
        fasta_lines.append(fasta_line)
    
    fasta_text = "\n".join(fasta_lines)

    write_text_file(path, fasta_text)

def read_cdhit_cluster_file(path):
    """
    Given a path to a CD-HIT cluster file, reads the file and returns a
    dictionary where the keys are the cluster IDs and the values are the
    cluster members.

    Args:
        path (str): The path to the CD-HIT cluster file to read.
    
    Returns:
        clusters (dict): A dictionary where the keys are the cluster IDs and the
            values are the cluster members
    """
    clusters_text = read_text_file(path).strip()
    cluster_entries = clusters_text[1:].split("\n>")
    clusters = dict()
    for cluster_entry in cluster_entries:
        cluster_entry_lines = cluster_entry.strip().split("\n")

        # Extract the cluster id from the header.
        cluster_header_line = cluster_entry_lines[0]
        cluster_id = int(cluster_header_line.strip().split(" ")[1])

        # Extract the cluster members.
        cluster_member_lines = cluster_entry_lines[1:]
        cluster_members = []
        for cluster_member_line in cluster_member_lines:
            member_length, member_entry = \
                cluster_member_line.strip().split(", >")
            member_id, _ = member_entry.split("...")
            cluster_members.append(member_id)

        clusters[cluster_id] = cluster_members
    
    return clusters

def chain_num_to_chain_id(chain_num):
    """
    Given a number chain_num, converts the number to a chain ID of letters.
    This uses "reverse spreadsheet style":
      0, 1, ...
      A, B, ..., Z, AA, BA, CA, ..., ZA, AB, BB, CB, ..., ZB, ...

    Args:
        chain_num (int): The number to convert to a chain ID. i starts at 0.
    
    Returns:
        chain_id (str): The chain ID corresponding to the number.
    """
    alphabet_length = 26
    
    # This algorithm is similar to converting to base 26, but we need to
    # subtract 1 from the number since mapping A to 0 base 26 results in some
    # issues (e.g. if A = 0 base 26, then AA = 00 base 26, which is not 
    # correct).
    chain_letter_list = []
    while chain_num >= 0:
        chain_letter_list.append(chr(ord("A") + (chain_num % alphabet_length))) 
        chain_num = (chain_num // 26) - 1

    chain_id = "".join(chain_letter_list)
    return chain_id

def compute_human_readable_ppm(specificity_data,
                               ppm_na_mpnn_format_key,
                               ppm_polymer_type,
                               method):
    """
    Converts the PPM to a human-readable format.

    Args:
        specificity_data (dict): The specificity data dictionary.
        ppm_na_mpnn_format_key (str): The key for the PPM in the specificity
            data dictionary.
        ppm_polymer_type (str): The polymer type of the PPM.
        method (str): The method used for specificity prediction.
    
    Returns:
        ppm (np.ndarray): The PPM in a human-readable format.
    """
    ppm_na_mpnn_format = specificity_data[ppm_na_mpnn_format_key]
    mask = specificity_data["mask"]
    if ppm_polymer_type == "dna":
        polymer_mask = specificity_data["dna_mask"]
        restypes = NAConstants.deep_pbs_restypes
    elif ppm_polymer_type == "rna":
        if method == "deeppbs":
            raise ValueError("DeepPBS does not support RNA specificity prediction.")
        polymer_mask = specificity_data["rna_mask"]
        restypes = NAConstants.rna_restypes
    else:
        raise ValueError(f"Invalid polymer type: {ppm_polymer_type}")
    
    # Convert the predicted PPM to the appropriate format.
    ppm = ppm_na_mpnn_format[np.logical_and(mask == 1, polymer_mask == 1)]
    ppm = ppm[:, [NAConstants.na_mpnn_restype_to_int[restype] for restype in restypes]]

    return ppm

def compute_human_readable_true_sequence(specificity_data,
                                         true_sequence_na_mpnn_format_key,
                                         ppm_polymer_type,
                                         method):
    """
    Converts the true sequence to a human-readable format.

    Args:
        specificity_data (dict): The specificity data dictionary.
        true_sequence_na_mpnn_format_key (str): The key for the true sequence
            in the specificity data dictionary.
        ppm_polymer_type (str): The polymer type of the PPM.
        method (str): The method used for specificity prediction.
    
    Returns:
        true_sequence (list): The true sequence in a human-readable format.
    """
    true_sequence_na_mpnn_format = specificity_data[true_sequence_na_mpnn_format_key]
    mask = specificity_data["mask"]
    if ppm_polymer_type == "dna":
        polymer_mask = specificity_data["dna_mask"]
        restype_to_int = NAConstants.deep_pbs_restype_to_int
    elif ppm_polymer_type == "rna":
        if method == "deeppbs":
            raise ValueError("DeepPBS does not support RNA specificity prediction.")
        polymer_mask = specificity_data["rna_mask"]
        restype_to_int = NAConstants.rna_restype_to_int
    else:
        raise ValueError(f"Invalid polymer type: {ppm_polymer_type}")
    
    # Convert the true sequence to the appropriate format.
    true_sequence = true_sequence[np.logical_and(mask == 1, polymer_mask == 1)]
    true_sequence = list(map(lambda restype_int:
        restype_to_int[NAConstants.na_mpnn_int_to_restype[restype_int]],
        true_sequence_na_mpnn_format))

    return true_sequence

################################################################################
# Constants
################################################################################
class NAConstants:
    # 1 letter codes for RNA residues.
    rna_restypes = [
        "A",
        "C",
        "G",
        "U",
    ]
    rna_restype_to_int = dict(zip(rna_restypes, range(len(rna_restypes))))

    # Unknown residues.
    rna_unknown_restype = "X"
    dssr_unknown_restype = "?"

    # Chain break characters.
    chain_break_character = "/"
    dssr_chain_break_character = "&"

    # DSSR represents modifications of residues with the lower case of their
    # base residue.
    dssr_modified_restypes = [rna_restype.lower() for rna_restype in rna_restypes]

    # NA-MPNN RNA residue type mapping.
    na_mpnn_rna_restype_to_rna_restype = {
        "b": "A",
        "d": "C",
        "h": "G",
        "u": "U",
        "y": "X"
    }

    # NA-MPNN na shared token representation.
    na_mpnn_na_shared_tokens = True

    # NA-MPNN residue type ordering.
    na_mpnn_restypes = [
        'ALA',
        'ARG',
        'ASN',
        'ASP',
        'CYS',
        'GLN',
        'GLU',
        'GLY',
        'HIS',
        'ILE',
        'LEU',
        'LYS',
        'MET',
        'PHE',
        'PRO',
        'SER',
        'THR',
        'TRP',
        'TYR',
        'VAL',
        'UNK',
        'DA',
        'DC',
        'DG',
        'DT',
        'DX',
        'A',
        'C',
        'G',
        'U',
        'RX',
        'MAS',
        'PAD'
    ]

    # NA-MPNN residue type to int mapping.
    na_mpnn_restype_to_int = dict(zip(na_mpnn_restypes, range(len(na_mpnn_restypes))))
    na_mpnn_int_to_restype = dict(zip(range(len(na_mpnn_restypes)), na_mpnn_restypes))

    if na_mpnn_na_shared_tokens:
        na_mpnn_restype_to_int["A"] = na_mpnn_restype_to_int["DA"]
        na_mpnn_restype_to_int["C"] = na_mpnn_restype_to_int["DC"]
        na_mpnn_restype_to_int["G"] = na_mpnn_restype_to_int["DG"]
        na_mpnn_restype_to_int["U"] = na_mpnn_restype_to_int["DT"]
        na_mpnn_restype_to_int["RX"] = na_mpnn_restype_to_int["DX"]
    
    # DeepPBS restype ordering.
    deep_pbs_restypes = [
        "DA",
        "DC",
        "DG",
        "DT"
    ]

    # DeepPBS restype to int mapping.
    deep_pbs_restype_to_int = dict(zip(deep_pbs_restypes, range(len(deep_pbs_restypes))))
    deep_pbs_int_to_restype = dict(zip(range(len(deep_pbs_restypes)), deep_pbs_restypes))

    # Min overlap length for ppm alignment.
    min_overlap_length = 5

    # 2D structure symbols for RNA.
    pair_symbols_list = [
        ("(", ")"),
        ("[", "]"),
        ("{", "}"),
        ("<", ">"),
        ("A", "a"),
        ("B", "b"),
        ("C", "c"),
        ("D", "d"),
        ("E", "e"),
        ("E", "e"),
        ("F", "f"),
        ("G", "g"),
        ("H", "h"),
        ("I", "i"),
        ("J", "j"),
        ("K", "k"),
        ("L", "l"),
        ("M", "m"),
        ("N", "n"),
        ("O", "o"),
        ("P", "p"),
        ("Q", "q"),
        ("R", "r"),
        ("S", "s"),
        ("T", "t"),
        ("U", "u"),
        ("V", "v"),
        ("W", "w"),
        ("X", "x"),
        ("Y", "y"),
        ("Z", "z"),
    ]

    # Create lists of open, close, and loop symbols.
    open_symbols = [pair_symbols[0] for pair_symbols in pair_symbols_list]
    close_symbols = [pair_symbols[1] for pair_symbols in pair_symbols_list]
    loop_symbols = [".", ","]

    # Create dictionaries to map open symbols to close symbols and vice versa.
    open_to_close = {pair_symbols[0]: pair_symbols[1] for pair_symbols in pair_symbols_list}
    close_to_open = {pair_symbols[1]: pair_symbols[0] for pair_symbols in pair_symbols_list}

################################################################################
# Sequence and Structure Standardization
################################################################################
def check_rna_sequence_validity(sequence, 
                                unknown_residue_allowed,
                                chain_breaks_allowed):
    """
    Given an rna sequence, checks the validity of the sequence.

    Args:
        sequence (str): The RNA sequence to check.
        unknown_residue_allowed (bool): Whether unknown residues are allowed in
            the sequence.
        chain_breaks_allowed (bool): Whether chain breaks are allowed in the
            sequence.
    
    Side Effects:
        Raises a ValueError if the sequence is invalid.
    """
    for c in sequence:
        if c in NAConstants.rna_restype_to_int:
            continue
        elif unknown_residue_allowed and c == NAConstants.rna_unknown_restype:
            continue
        elif chain_breaks_allowed and c == NAConstants.chain_break_character:
            continue
        else:
            raise ValueError(f"Invalid character in sequence: {c}")   

def standardize_rna_sequence(sequence, 
                             method = None,
                             remove_chain_breaks = False):
    """
    Given an RNA sequence, standardizes the sequence to a canonical form.

    NOTE: This method is only intended for use with RNA sequences.

    Args:
        sequence (str): The RNA sequence to standardize.
        method (str): The method to use for standardization.
            Options:
                "na_mpnn": Standardize the sequence using the NA-MPNN RNA
                    residue type mapping.
                "dssr": Standardize the sequence using the DSSR unknown
                    residue and chain break characters.
                None: no standardization.
        remove_chain_breaks (bool): Whether to remove chain breaks from the
            sequence. 
            NOTE: This option should only be True if the user is certain that 
                the sequence does not contain any chain breaks and that the
                presence of any chain breaks is an error.
    
    Returns:
        standard_sequence (str): The standardized RNA sequence.
    """
    standard_sequence = []

    # Standardize the sequence.
    for c in sequence:
        # Convert the bdhuy characters from NA-MPNN to ACGUX.
        if method == "na_mpnn" and \
           c in NAConstants.na_mpnn_rna_restype_to_rna_restype:
            standard_sequence.append(NAConstants.na_mpnn_rna_restype_to_rna_restype[c])
        # Standardize the dssr unknown residue.
        elif method == "dssr" and c == NAConstants.dssr_unknown_restype:
            standard_sequence.append(NAConstants.rna_unknown_restype)
        # Standardize the dssr chain break character.
        elif method == "dssr" and c == NAConstants.dssr_chain_break_character:
            standard_sequence.append(NAConstants.chain_break_character)
        # DSSR represents modifications of residues with the lower case of their
        # base residue. We convert them to the unknown residue.
        elif method == "dssr" and c in NAConstants.dssr_modified_restypes:
            standard_sequence.append(NAConstants.rna_unknown_restype)
        else:
            standard_sequence.append(c)
            
    # Remove chain breaks if specified.
    if remove_chain_breaks:
        standard_sequence = [c for c in standard_sequence if c != NAConstants.chain_break_character]
    
    standard_sequence = "".join(standard_sequence)

    # Check the validity of the standard sequence.
    check_rna_sequence_validity(standard_sequence, 
                                unknown_residue_allowed = True,
                                chain_breaks_allowed = True)

    return standard_sequence

def check_secondary_structure_validity(secondary_structure):
    """
    Given a secondary structure string, checks the validity of the secondary
    structure string. 

    Args:
        secondary_structure (str): The secondary structure string.
    
    Side Effects:
        Raises a ValueError if the secondary structure string is invalid.
    """
    calculate_base_pairs_and_loops_from_secondary_structure(secondary_structure)

def standardize_secondary_structure(secondary_structure,
                                    method = None,
                                    replace_unknown_restypes = False,
                                    remove_chain_breaks = False):
    """
    Given a secondary structure string, standardizes the secondary structure
    to a canonical form.

    NOTE: This method is only intended for use with NA secondary structure.

    Args:
        secondary_structure (str): The secondary structure string to 
            standardize.
        method (str): The method to use for standardization.
            Options:
                "dssr": Standardize the secondary structure using the DSSR
                    unknown residue and chain break characters.
                None: no standardization.
        replace_unknown_restypes (bool): Whether to replace unknown residues
            with loop symbols in the secondary structure. This option is only
            valid if method is "dssr". This option should only be True if the
            user is certain that the secondary structure does not contain any
            unknown residues and that the presence of any unknown residues is an
            error.
        remove_chain_breaks (bool): Whether to remove chain breaks from the
            secondary structure. This option is only valid if method is "dssr".
            This option should only be True if the user is certain that the
            secondary structure does not contain any chain breaks and that the
            presence of any chain breaks is an error.
    """
    standard_secondary_structure = []

    # Standardize the secondary structure.
    for c in secondary_structure:
        if method == "dssr" and \
           replace_unknown_restypes and \
           c == NAConstants.dssr_unknown_restype:
            standard_secondary_structure.append(NAConstants.loop_symbols[0])
        elif method == "dssr" and \
             remove_chain_breaks and \
             c == NAConstants.dssr_chain_break_character:
            continue
        else:
            standard_secondary_structure.append(c)
    
    standard_secondary_structure = "".join(standard_secondary_structure)

    # Check the validity of the standard secondary structure.
    check_secondary_structure_validity(standard_secondary_structure)

    return standard_secondary_structure

################################################################################
# Structure to Sequence and Secondary Structure
################################################################################
def run_dssr(structure_path, 
             dssr_path = "/projects/ml/afavor/software/dssr-x3dna/x3dna-dssr"):
    """
    Given a path to a tertiary structure file containing nucleic acid, runs the
    DSSR algorithm to extract the nucleic acid sequence and determine the
    nucleic acid secondary structure.

    Args:
        structure_path (str): The path to the tertiary structure file.
        dssr_path (str): The path to the DSSR executable.
    
    Returns:
        result (dict): A dictionary containing:
            sequence (str): The nucleic acid sequence from the tertiary 
                structure.
            secondary_structure (str): The nucleic acid secondary structure from 
                the tertiary structure.
    """
    # Turn the structure_path into an absolute path.
    structure_path = os.path.abspath(structure_path)

    # Check that the structure_path exists.
    if not os.path.exists(structure_path):
        raise ValueError(f"Invalid structure path: {structure_path}")

    # Get the file name of the structure path (removing extension).
    structure_name = os.path.splitext(os.path.basename(structure_path))[0]

    # Create a temporary directory for the outputs, and ensure it gets removed
    # on script exit.
    tmp_directory = tempfile.TemporaryDirectory()

    # Compute the paths for the output files.
    out_path = os.path.join(tmp_directory.name, f"{structure_name}.out")
    dbn_path = os.path.join(tmp_directory.name, f"{structure_name}-2ndstrs.dbn")

    # Run the DSSR algorithm.
    try:
        subprocess.run(
            [
                str(dssr_path),
                f"-i={structure_path}",
                f"-o={out_path}",
                f"--prefix={structure_name}"
            ], 
            check = True,
            cwd = tmp_directory.name,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        # Read the dbn file.
        dbn_text = read_text_file(dbn_path)

        # Extract the sequence string.
        sequence = dbn_text.split("\n")[1]

        # Extract the secondary structure string.
        secondary_structure = dbn_text.split("\n")[2]

        tmp_directory.cleanup()

        result = {
            "sequence": sequence,
            "secondary_structure": secondary_structure
        }

        return result
    except subprocess.CalledProcessError as e:
        tmp_directory.cleanup()
        raise e

################################################################################
# Sequence to Predicted Secondary Structure and Reactivity Profile
################################################################################
def run_eternafold(sequence,
                   eternafold_path = "/projects/ml/afavor/software/EternaFold/src/contrafold"):
    """
    Given a sequence, run the EternaFold algorithm to predict the secondary
    structure of the sequence.

    Args:
        sequence (str): The sequence to predict the secondary structure for.
        eternafold_path (str): The path to the EternaFold executable.

    Returns:
        result (dict): A dictionary containing:
            predicted_secondary_structure (str): The predicted secondary 
                structure of the sequence.
    """
    # Check that the RNA sequence is valid.
    check_rna_sequence_validity(sequence, 
                                unknown_residue_allowed = False, 
                                chain_breaks_allowed = False)

    # Create the input and output files for EternaFold.
    eternafold_input_file = tempfile.NamedTemporaryFile(mode = "wt")
    eternafold_output_file = tempfile.NamedTemporaryFile(mode = "wt")

    # Write the sequence to the input file.
    eternafold_input_file.write(sequence)
    eternafold_input_file.flush()

    # Run EternaFold.
    try:
        subprocess.run(
            [
                str(eternafold_path),
                "predict",
                eternafold_input_file.name
            ],
            check = True,
            stdout = eternafold_output_file,
            stderr = subprocess.DEVNULL
        )

        eternafold_output_text = read_text_file(eternafold_output_file.name)

        # Extract the predicted secondary structure from the EternaFold output.
        eternafold_output_lines = eternafold_output_text.strip().split("\n")

        # The predicted secondary structure is the last line of the output.
        predicted_secondary_structure = eternafold_output_lines[-1]

        eternafold_input_file.close()
        eternafold_output_file.close()

        result = {
            "predicted_secondary_structure": predicted_secondary_structure
        }

        return result
    except (subprocess.CalledProcessError, ValueError) as e:
        eternafold_input_file.close()
        eternafold_output_file.close()
        raise e

def run_ribonanza_net_reactivity_profile(sequence,
                                         batch_size = 1,
                                         ribonanza_net_apptainer_path = "/software/containers/users/afavor/PPI_design_mpnn.sif",
                                         ribonanza_net_path = "/home/akubaney/projects/na_mpnn/evaluation/run_ribonanza_net.py",):
    """
    Given a sequence, runs the RibonanzaNet algorithm to predict the reactivity
    profile of the sequence.

    Args:
        sequence (str): The sequence to predict the reactivity profile for.
        batch_size (int): The number of samples to predict in a batch.
        ribonanza_net_apptainer_path (str): The path to the RibonanzaNet
            apptainer for running RibonanzaNet.
        ribonanza_net_path (str): The path to the RibonanzaNet run file.
    
    Returns:
        result (dict): A dictionary containing:
            predicted_2A3_reactivity_profiles (list of float lists): A list of
                predicted reactivity profiles of the sequence for the 2A3 probe.
            predicted_DMS_reactivity_profiles (list of float lists): A list of
                predicted reactivity profiles of the sequence for the DMS probe.
    """    
    # Check that the RNA sequence is valid.
    check_rna_sequence_validity(sequence,
                                unknown_residue_allowed = False,
                                chain_breaks_allowed = False)
    
    # Create a temporary directory for the outputs, and ensure it gets removed
    # on script exit.
    tmp_directory = tempfile.TemporaryDirectory()

    # Compute the paths for the output files.
    out_path = os.path.join(tmp_directory.name, "output.npy")

    # Run the RibonanzaNet algorithm to predict the reactivity profile.
    try:
        subprocess.run(
            [
                "apptainer",
                "exec",
                str(ribonanza_net_apptainer_path),
                "python",
                str(ribonanza_net_path),
                "reactivity_profile",
                str(sequence),
                str(tmp_directory.name),
                str(batch_size)
            ], 
            check = True,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        # Read the output file.
        out_dict = np.load(out_path, allow_pickle = True).item()

        # Extract the predicted reactivity profiles.
        result = {
            "predicted_2A3_reactivity_profiles": out_dict["predicted_2A3_reactivity_profiles"],
            "predicted_DMS_reactivity_profiles": out_dict["predicted_DMS_reactivity_profiles"]
        }

        # Clean up the temporary directory.
        tmp_directory.cleanup()

        return result
    except subprocess.CalledProcessError as e:
        tmp_directory.cleanup()
        raise e

def run_ribonanza_net_secondary_structure(sequence,
                                          batch_size = 1,
                                          ribonanza_net_apptainer_path = "/software/containers/users/afavor/PPI_design_mpnn.sif",
                                          ribonanza_net_path = "/home/akubaney/projects/na_mpnn/evaluation/run_ribonanza_net.py"):
    """
    Given a sequence, runs the RibonanzaNet algorithm to predict the secondary
    structure of the sequence.

    Args:
        sequence (str): The sequence to predict the secondary structure for.
        batch_size (int): The number of samples to predict in a batch.
        ribonanza_net_apptainer_path (str): The path to the RibonanzaNet
            apptainer for running RibonanzaNet.
        ribonanza_net_path (str): The path to the RibonanzaNet run file.
    
    Returns:
        result (dict): A dictionary containing:
            predicted_secondary_structures (str list): The predicted secondary
                structures of the sequence.
    """    
    # Check that the RNA sequence is valid.
    check_rna_sequence_validity(sequence,
                                unknown_residue_allowed = False,
                                chain_breaks_allowed = False)
    
    # Create a temporary directory for the outputs, and ensure it gets removed
    # on script exit.
    tmp_directory = tempfile.TemporaryDirectory()

    # Compute the paths for the output files.
    out_path = os.path.join(tmp_directory.name, "output.npy")

    # Run the RibonanzaNet algorithm to predict the secondary structure.
    try:
        subprocess.run(
            [
                "apptainer",
                "exec",
                str(ribonanza_net_apptainer_path),
                "python",
                str(ribonanza_net_path),
                "secondary_structure",
                str(sequence),
                str(tmp_directory.name),
                str(batch_size)
            ], 
            check = True,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        # Read the output file.
        out_dict = np.load(out_path, allow_pickle = True).item()

        # Extract the predicted secondary structures.
        result = {
            "predicted_secondary_structures": out_dict["predicted_secondary_structures"]
        }

        # Clean up the temporary directory.
        tmp_directory.cleanup()

        return result
    except subprocess.CalledProcessError as e:
        tmp_directory.cleanup()
        raise e

################################################################################
# Sequence to Predicted Structure
################################################################################
def run_alphafold3(name, 
                   sequences_and_polytypes,
                   output_dir,
                   num_diffusion_samples = 5,
                   num_seeds = 1,
                   fixed_seeds = None,
                   run_data_pipeline = False,
                   buckets = "1",
                   alphafold3_apptainer_path = "/software/containers/users/ikalvet/mlfold3/mlfold3_01.sif",
                   alphafold3_path = "/opt/alphafold3/run_alphafold.py",
                   model_dir = "/databases/alphafold",):
    """
    Given a name, a list of sequences and polytypes, and an output directory,
    runs AlphaFold3 to predict the structure of the complex.

    Args:
        name (str): A name of the complex.
        sequences_and_polytypes ((str, str) list): A list of tuples, where each
            tuple contains the sequence and polytype of a sequence.
        output_dir (str): The path to the output directory.
        num_diffusion_samples (int): The number of diffusion samples to 
            generate. Default is 5.
        num_seeds (int): The number of model seeds to generate and use. Default
            is 1. This argument is mutually exclusive with fixed_seeds.
        fixed_seeds (int list): A list of fixed seeds to use for the model. This
            argument is mutually exclusive with num_seeds.
        run_data_pipeline (bool): Whether to run the data pipeline (whether to
            perform the MSA and templates searches).
        buckets (str): A comma separated list of integers. Strictly increasing 
            order of token sizes for which to cache compilations. For any input 
            with more tokens than the largest bucket size, a new bucket is 
            created for exactly that number of tokens. The "1" bucket is a 
            trick to ensure no padding occurs; although if running batches could
            cause a lot of model recompilation. The alphafold3 default is
            "256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120".
        alphafold3_apptainer_path (str): The path to the AlphaFold3 apptainer
            for running AlphaFold3.
        alphafold3_path (str): The path to the AlphaFold3 run file.
        model_dir (str): The path to the AlphaFold3 model directory.
    
    Returns:
        result (dict): A dictionary containing:
            json_input_path (str): The path to the input JSON file.
            predicted_structure_path (str): The path to the predicted structure
                file.
            predicted_confidences_path (str): The path to the predicted
                confidences file.
            summary_confidences_path (str): The path to the summary confidences
                file.
            ptm (float): The predicted PTM score.
            plddt (float): The predicted pLDDT score.
            pae (float): The predicted pAE score.
    """
    # Check that both num_seeds and fixed_seeds are not set.
    if num_seeds is not None and fixed_seeds is not None:
        raise ValueError("Both num_seeds and fixed_seeds cannot be set at the same time.")

    # If the output directory for the specified name already exists,
    # raise an error.
    name_output_directory = os.path.join(output_dir, name)
    if os.path.exists(name_output_directory):
        raise ValueError(f"Output directory already exists: {name_output_directory}")
    
    # Prepare the model seed input.
    if fixed_seeds is not None:
        model_seeds = fixed_seeds
    else:
        # Generate random seeds.
        seed_rng = np.random.default_rng()
        model_seeds = [int(seed_rng.integers(0, 2 ** 32 - 1)) for i in range(num_seeds)]

    # Prepare the sequences input, with no MSA or template.
    sequences_input = []
    for i, (sequence, polytype) in enumerate(sequences_and_polytypes):
        sequences_entry_dict = {
            polytype: {
                "id": chain_num_to_chain_id(i),
                "sequence": sequence,
                "unpairedMsa": ""
            }
        }
        sequences_input.append(sequences_entry_dict)

    alphafold3_input_json_dict = {
        "dialect": "alphafold3",
        "version": 3,
        "name": name,
        "modelSeeds": model_seeds,
        "sequences": sequences_input
    }
    
    # Set up the input JSON file.
    temp_json_file = tempfile.NamedTemporaryFile(mode = "wt", suffix = ".json")

    # Write the input JSON file.
    write_json_file(temp_json_file.name, alphafold3_input_json_dict)

    # Run AlphaFold3.
    try:
        subprocess.run(
            [
                alphafold3_apptainer_path,
                "python",
                alphafold3_path,
                f"--model_dir={model_dir}",
                f"--run_data_pipeline={run_data_pipeline}",
                f"--buckets={buckets}",
                f"--num_diffusion_samples={num_diffusion_samples}",
                f"--output_dir={output_dir}",
                f"--json_path={temp_json_file.name}",
            ],
            check = True
        )
    except (subprocess.CalledProcessError, ValueError) as e:
        temp_json_file.close()
        raise e 

    # Close the temporary file.
    temp_json_file.close()

    # Process the outputs.
    json_input_path = os.path.join(name_output_directory, f"{name}_data.json")
    predicted_structure_path = os.path.join(name_output_directory, f"{name}_model.cif")
    predicted_confidences_path = os.path.join(name_output_directory, f"{name}_confidences.json")
    summary_confidences_path = os.path.join(name_output_directory, f"{name}_summary_confidences.json")

    # Check that the output files exist.
    if not os.path.exists(json_input_path):
        raise ValueError(f"Output JSON file not found: {json_input_path}")
    if not os.path.exists(predicted_structure_path):
        raise ValueError(f"Predicted structure file not found: {predicted_structure_path}")
    if not os.path.exists(predicted_confidences_path):
        raise ValueError(f"Predicted confidences file not found: {predicted_confidences_path}")
    if not os.path.exists(summary_confidences_path):
        raise ValueError(f"Summary confidences file not found: {summary_confidences_path}")
    
    # Extract confidence scores.
    summary_confidences_dict = read_json_file(summary_confidences_path)
    ptm = summary_confidences_dict["ptm"]

    predicted_confidences_dict = read_json_file(predicted_confidences_path)

    atom_plddts = predicted_confidences_dict["atom_plddts"]
    plddt = np.mean(atom_plddts)

    pae_matrix = predicted_confidences_dict["pae"]
    pae = np.mean(pae_matrix)

    result = {
        "json_input_path": json_input_path,
        "predicted_structure_path": predicted_structure_path,
        "predicted_confidences_path": predicted_confidences_path,
        "summary_confidences_path": summary_confidences_path,
        "ptm": ptm,
        "plddt": plddt,
        "pae": pae
    }
    
    return result

################################################################################
# Specificity Prediction
################################################################################    
def run_na_mpnn_specificity(structure_path,
                            output_directory = None,
                            batch_size = 1,
                            number_of_batches = 1,
                            temperature = 0.1,
                            omit_AA = "",
                            design_na_only = 0,
                            load_residues_with_missing_atoms = 0,
                            output_pdbs = 0,
                            output_sequences = 0,
                            output_specificity = 1,
                            catch_failed_inferences = 1,
                            na_mpnn_apptainer_path = "/software/containers/mlfold.sif",
                            na_mpnn_path = "/home/akubaney/projects/fused_mpnn/run.py",
                            na_mpnn_model_path = None):
    """
    Given a path to a tertiary structure file, runs the NA-MPNN algorithm to
    predict the nucleic acid specificity of the structure.

    Args:
        structure_path (str): The path to the tertiary structure file.
        output_directory (str): The path to the output directory. If None, a
            temporary directory will be created.
        batch_size (int): The number of samples to predict in a batch.
        number_of_batches (int): The number of batches to predict.
        temperature (float): The temperature for sampling.
        omit_AA (str): The residues to omit from the prediction.
        design_na_only (int): Whether to design only nucleic acids.
        load_residues_with_missing_atoms (int): Whether to load residues with
            missing atoms.
        output_pdbs (int): Whether to output the PDB files.
        output_sequences (int): Whether to output the sequences.
        output_specificity (int): Whether to output the specificity.
        catch_failed_inferences (int): Whether to catch failed inferences.
        na_mpnn_apptainer_path (str): The path to the NA-MPNN apptainer for
            running NA-MPNN.
        na_mpnn_path (str): The path to the NA-MPNN run file.
        na_mpnn_model_path (str): The path to the NA-MPNN model weights file.
    
    Returns:
        result (dict): A dictionary containing:
            input_structure_name (str): The name of the input structure.
            input_structure_path (str): The path to the input structure.
            name (str): The name of the input structure.
            predicted_ppm_na_mpnn_format (np.ndarray): The predicted PPM in the
                NA-MPNN format.
            true_sequence_na_mpnn_format (np.ndarray): The true sequence in the
                NA-MPNN format.
            chain_labels (np.ndarray): The chain labels.
            mask (np.ndarray): The mask for the structure; which residues have
                all backbone atoms.
            protein_mask (np.ndarray): The mask for the protein residues.
            dna_mask (np.ndarray): The mask for the DNA residues.
            rna_mask (np.ndarray): The mask for the RNA residues.
            encoded_residues (str list): The string names of the residues.
            encoded_residues_dict (dict): The dictionary mapping the string
                names of the residues to their indices.
            specificity_method (str): The method used for specificity
                prediction.
            model_weights_path (str): The path to the model weights file.
            num_samples (int): The number of samples predicted.
            temperature (float): The temperature used for sampling.            
    """
    # Convert the structure path to an absolute path.
    structure_path = os.path.abspath(structure_path)

    # Check that the structure path exists.
    if not os.path.exists(structure_path):
        raise ValueError(f"Invalid structure path: {structure_path}")

    # If the output directory is not specified, create a temporary directory.
    # The temporary directory will be automatically cleaned up when the script
    # exits.
    if output_directory is None:
        tmp_directory = tempfile.TemporaryDirectory()
        output_directory = tmp_directory.name
    else:
        output_directory = os.path.abspath(output_directory)
    
    # Compute the output directory for the specificity.
    specificity_output_directory = os.path.join(output_directory, "specificity")
    
    # Compute the name of the structure.
    structure_name = os.path.splitext(os.path.basename(structure_path))[0]

    # Run the NA-MPNN specificity prediction algorithm.
    try:
        subprocess.run(
            [
                "apptainer",
                "exec",
                na_mpnn_apptainer_path,
                "python",
                na_mpnn_path,
                "--model_type",
                str("na_mpnn"),
                "--checkpoint_na_mpnn",
                str(na_mpnn_model_path),
                "--pdb_path",
                str(structure_path),
                "--out_folder",
                str(output_directory),
                "--number_of_batches",
                str(number_of_batches),
                "--batch_size",
                str(batch_size),
                "--temperature",
                str(temperature),
                "--omit_AA",
                str(omit_AA),
                "--design_na_only",
                str(design_na_only),
                "--load_residues_with_missing_atoms",
                str(load_residues_with_missing_atoms),
                "--output_pdbs",
                str(output_pdbs),
                "--output_sequences",
                str(output_sequences),
                "--output_specificity",
                str(output_specificity),
                "--catch_failed_inferences",
                str(catch_failed_inferences)                
            ],
            check = True,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        # Check that the output specificity file exists.
        specificity_path = os.path.join(specificity_output_directory, f"{structure_name}.npz")

        if not os.path.exists(specificity_path):
            raise ValueError(f"Output specificity file not found: {specificity_path}")
        
        # Read the output specificity file.
        specificity_dict = np.load(specificity_path, allow_pickle = True)

        # Extract the predicted PPM, true sequence, chain labels, and masks.
        result = {
            "input_structure_name": structure_name,
            "input_structure_path": structure_path,
            "name": structure_name,
            "predicted_ppm_na_mpnn_format": specificity_dict["predicted_ppm"],
            "true_sequence_na_mpnn_format": specificity_dict["true_sequence"],
            "chain_labels": specificity_dict["chain_labels"],
            "mask": specificity_dict["mask"],
            "protein_mask": specificity_dict["protein_mask"],
            "dna_mask": specificity_dict["dna_mask"],
            "rna_mask": specificity_dict["rna_mask"],
            "encoded_residues": specificity_dict["encoded_residues"],
            "encoded_residues_dict": specificity_dict["encoded_residues_dict"],
            "specificity_method": "na_mpnn",
            "model_weights_path": na_mpnn_model_path,
            "num_samples": number_of_batches * batch_size,
            "temperature": temperature,
        }

        # Clean up the temporary directory if it was created.
        if output_directory is None:
            tmp_directory.cleanup()
        
        return result
    except (subprocess.CalledProcessError, ValueError) as e:
        # Clean up the temporary directory if it was created.
        if output_directory is None:
            tmp_directory.cleanup()
        raise e

def run_deeppbs(structure_path,
                output_directory = None,
                deep_pbs_apptainer_path = "/software/containers/users/akubaney/deeppbs.sif",
                deep_pbs_directory = "/home/akubaney/software/DeepPBS/"):
    """
    Given a path to a tertiary structure file, runs the DeepPBS algorithm to
    predict the nucleic acid specificity of the structure.

    Args:
        structure_path (str): The path to the tertiary structure file.
        output_directory (str): The path to the output directory. If None, a
            temporary directory will be created.
        deep_pbs_apptainer_path (str): The path to the DeepPBS apptainer for
            running DeepPBS.
        deep_pbs_directory (str): The path to the DeepPBS directory.
    
    Returns:
        result (dict): A dictionary containing:
            input_structure_name (str): The name of the input structure.
            input_structure_path (str): The path to the input structure.
            name (str): The name of the input structure.
            predicted_ppm_na_mpnn_format (np.ndarray): The predicted PPM in the
                NA-MPNN format.
            true_sequence_na_mpnn_format (np.ndarray): The true sequence in the
                NA-MPNN format.
            chain_labels (np.ndarray): The chain labels.
            mask (np.ndarray): The mask for the structure; which residues have
                all backbone atoms.
            protein_mask (np.ndarray): The mask for the protein residues.
            dna_mask (np.ndarray): The mask for the DNA residues.
            rna_mask (np.ndarray): The mask for the RNA residues.
            encoded_residues (str list): The string names of the residues.
            encoded_residues_dict (dict): The dictionary mapping the string
                names of the residues to their indices.
            specificity_method (str): The method used for specificity
                prediction.
            model_weights_path (str): The path to the model weights file.
            num_samples (int): The number of samples predicted.
            temperature (float): The temperature used for sampling.
    """
    # Convert the structure path to an absolute path.
    structure_path = os.path.abspath(structure_path)

    # Check that the structure path exists.
    if not os.path.exists(structure_path):
        raise ValueError(f"Invalid structure path: {structure_path}")
    
    # If the output directory is not specified, create a temporary directory.
    # The temporary directory will be automatically cleaned up when the script
    # exits.
    if output_directory is None:
        tmp_directory = tempfile.TemporaryDirectory()
        output_directory = tmp_directory.name
    else:
        output_directory = os.path.abspath(output_directory)
    
    # Compute the name of the structure.
    structure_name, extension = os.path.splitext(os.path.basename(structure_path))

    # Compute the output directory for the specificity.
    specificity_output_directory = os.path.join(output_directory, "specificity")

    # Create the output directory if it does not exist.
    if not os.path.exists(specificity_output_directory):
        os.makedirs(specificity_output_directory)

    # Create a temporary directory for handling intermediate outputs.
    tmp_intermediate_directory = tempfile.TemporaryDirectory()

    # Create the pdb directory for the tool, and copy the structure file into
    # the directory.
    tmp_pdb_directory = os.path.join(tmp_intermediate_directory.name, "pdb")
    os.makedirs(tmp_pdb_directory)
    shutil.copy(structure_path, tmp_pdb_directory)

    # Create the input text file for featurization, and write the pdb file name
    # into the input text file.
    tmp_input_text_file = os.path.join(tmp_intermediate_directory.name, "input.txt")
    write_text_file(tmp_input_text_file, f"{structure_name}{extension}")

    # Create the npz directory for the featurized output.
    tmp_npz_directory = os.path.join(tmp_intermediate_directory.name, "npz")
    os.makedirs(tmp_npz_directory)

    # Create the input text file for prediction, and write the npz basename.
    tmp_predict_input_text_file = os.path.join(tmp_intermediate_directory.name, "predict_input.txt")
    write_text_file(tmp_predict_input_text_file, f"{structure_name}.npz")

    # Create the output directory for the prediction.
    tmp_output_directory = os.path.join(tmp_intermediate_directory.name, "output")
    os.makedirs(tmp_output_directory)

    # The file name for the output.
    tmp_output_file_name = os.path.join(tmp_output_directory, "npzs", f"{structure_name}.npz_predict.npz")

    # Run the DeepPBS algorithm.
    deep_pbs_featurize_path = os.path.join(deep_pbs_directory, "run", "process_co_crystal.py")
    deep_pbs_featurize_config_path = os.path.join(deep_pbs_directory, "run", "process", "process_config.json") 
    deep_pbs_predict_path = os.path.join(deep_pbs_directory, "run", "predict.py")
    deep_pbs_predict_config_path = os.path.join(deep_pbs_directory, "run", "process", "pred_configs", "pred_config_deeppbs.json")
    try:
        subprocess.run(
            [
                "apptainer",
                "exec",
                deep_pbs_apptainer_path,
                "python",
                deep_pbs_featurize_path,
                tmp_input_text_file,
                deep_pbs_featurize_config_path,
                "--no_pwm"
            ],
            check = True,
            cwd = tmp_intermediate_directory.name,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )
        subprocess.run(
            [
                "apptainer",
                "exec",
                deep_pbs_apptainer_path,
                "python",
                deep_pbs_predict_path,
                tmp_predict_input_text_file,
                tmp_output_directory,
                "-c",
                deep_pbs_predict_config_path
            ],
            check = True,
            cwd = tmp_intermediate_directory.name,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        # Check that the output file exists.
        if not os.path.exists(tmp_output_file_name):
            raise ValueError(f"Output file not found: {tmp_output_file_name}")
        
        # Read the output file.
        deep_pbs_dict = np.load(tmp_output_file_name, allow_pickle = True)

        # Load the predicted ppm and true sequence.
        deep_pbs_predicted_ppm = deep_pbs_dict["P"]
        deep_pbs_true_sequence_one_hot = deep_pbs_dict["Seq"]

        # Compute the base pairing ppm and sequence.
        deep_pbs_predicted_bp_ppm = np.flip(np.flip(deep_pbs_predicted_ppm, axis = 1), axis = 0)
        deep_pbs_bp_true_sequence_one_hot = np.flip(np.flip(deep_pbs_true_sequence_one_hot, axis = 1), axis = 0)

        # Concatenate the base pairing ppm and sequence, and ensure that the
        # chain labels array are different across the concatenation.
        chain_labels = np.concatenate(
            (
                np.zeros(deep_pbs_predicted_ppm.shape[0], dtype = np.int32),
                np.ones(deep_pbs_predicted_bp_ppm.shape[0], dtype = np.int32)
            ),
            axis = 0
        )
        deep_pbs_predicted_ppm = np.concatenate((deep_pbs_predicted_ppm, deep_pbs_predicted_bp_ppm), axis = 0)
        deep_pbs_true_sequence_one_hot = np.concatenate((deep_pbs_true_sequence_one_hot, deep_pbs_bp_true_sequence_one_hot), axis = 0)

        # Convert the predicted ppm to NA-MPNN format.
        predicted_ppm = np.zeros((deep_pbs_predicted_ppm.shape[0], len(NAConstants.na_mpnn_restype_to_int)), dtype = np.float64)
        predicted_ppm[:, NAConstants.na_mpnn_restype_to_int["DA"]] = deep_pbs_predicted_ppm[:, NAConstants.deep_pbs_restype_to_int["DA"]]
        predicted_ppm[:, NAConstants.na_mpnn_restype_to_int["DC"]] = deep_pbs_predicted_ppm[:, NAConstants.deep_pbs_restype_to_int["DC"]]
        predicted_ppm[:, NAConstants.na_mpnn_restype_to_int["DG"]] = deep_pbs_predicted_ppm[:, NAConstants.deep_pbs_restype_to_int["DG"]]
        predicted_ppm[:, NAConstants.na_mpnn_restype_to_int["DT"]] = deep_pbs_predicted_ppm[:, NAConstants.deep_pbs_restype_to_int["DT"]]

        # Convert the true sequence to NA-MPNN format.
        deep_pbs_true_sequence = np.argmax(deep_pbs_true_sequence_one_hot, axis = -1)
        true_sequence = list(map(lambda restype: 
            NAConstants.na_mpnn_restype_to_int[NAConstants.deep_pbs_int_to_restype[restype]],
            deep_pbs_true_sequence))

        # Extract the predicted PPM, true sequence, chain labels, and masks.
        result = {
            "input_structure_name": structure_name,
            "input_structure_path": structure_path,
            "name": structure_name,
            "predicted_ppm_na_mpnn_format": predicted_ppm,
            "true_sequence_na_mpnn_format": true_sequence,
            "chain_labels": chain_labels,
            "mask": np.ones(len(predicted_ppm), dtype = np.int32),
            "protein_mask": np.zeros(len(predicted_ppm), dtype = np.int32),
            "dna_mask": np.ones(len(predicted_ppm), dtype = np.int32),
            "rna_mask": np.zeros(len(predicted_ppm), dtype = np.int32),
            "encoded_residues": None,
            "encoded_residues_dict": None,
            "specificity_method": "deeppbs",
            "model_weights_path": None,
            "num_samples": 1,
            "temperature": None
        }

        # Save the output file to the output directory.
        output_file_name = os.path.join(specificity_output_directory, f"{structure_name}.npz")
        shutil.copy(tmp_output_file_name, output_file_name)

        # Clean up the temporary directories.
        tmp_intermediate_directory.cleanup()
        if output_directory is None:
            tmp_directory.cleanup()
        
        return result
    except (subprocess.CalledProcessError, ValueError) as e:
        # Clean up the temporary directories.
        tmp_intermediate_directory.cleanup()
        if output_directory is None:
            tmp_directory.cleanup()
        raise e

################################################################################
# Sequence Comparison
################################################################################
def calculate_sequence_recovery(reference_sequence, 
                                subject_sequence,
                                chain_breaks_allowed = False,
                                unknown_residue_allowed_in_reference = False):
    """
    Given a reference sequence and a subject sequence, calculates the sequence 
    recovery of the subject sequence.

    Args:
        reference_sequence (str): The reference sequence to calculate the
            sequence recovery against.
        subject_sequence (str): The sequence to calculate the sequence recovery 
            for.
        chain_breaks_allowed (bool): Whether chain breaks are allowed in the
            sequence.
        unknown_residue_allowed_in_reference (bool): Whether unknown residues
            are allowed in the reference sequence.
    
    Returns:
        result (dict): A dictionary containing:
            sequence_recovery (float): The sequence recovery of the sequence.
    """
    # Check that the subject sequence and reference sequence have the same 
    # length.
    if len(subject_sequence) != len(reference_sequence):
        raise ValueError(f"Length of subject sequence ({len(subject_sequence)}) must match length of reference sequence ({len(reference_sequence)}).")
    
    # Check the validity of the subject sequence.
    check_rna_sequence_validity(subject_sequence,
                                unknown_residue_allowed = False,
                                chain_breaks_allowed = chain_breaks_allowed)

    # Check the validity of the reference sequence.
    check_rna_sequence_validity(reference_sequence,
                                unknown_residue_allowed = unknown_residue_allowed_in_reference,
                                chain_breaks_allowed = chain_breaks_allowed)
    
    # Calculate the number of correct residues.
    num_correct = 0
    num_residues = 0
    for subject_residue, reference_residue in zip(subject_sequence, reference_sequence):
        # Skip unknown residues in the reference sequence.
        if unknown_residue_allowed_in_reference and \
           reference_residue == NAConstants.rna_unknown_restype:
            continue
        # Skip chain breaks if they occur in both sequences.
        elif chain_breaks_allowed and \
           (subject_residue == NAConstants.chain_break_character or \
            reference_residue == NAConstants.chain_break_character):
            if not (subject_residue == NAConstants.chain_break_character and \
                    reference_residue == NAConstants.chain_break_character):
                raise ValueError("Chain breaks must occur at the same position in both sequences.")
            continue
        else:
            num_residues += 1
            if subject_residue == reference_residue:
                num_correct += 1

    # Calculate the sequence recovery.
    if num_residues == 0:
        raise ValueError("Number of residues must be greater than 0.")
    
    sequence_recovery = num_correct / num_residues

    result = {
        "sequence_recovery": sequence_recovery
    }

    return result

################################################################################
# Secondary Structure and Reactivity Profile Comparison
################################################################################
def calculate_base_pairs_and_loops_from_secondary_structure(secondary_structure):
    """
    Given a secondary structure string, calculates the base pair and loop 
    indices. Note, this function can also be used to check the validity of
    secondary structure strings.

    Args:
        secondary_structure (str): The secondary structure string.
    
    Returns:
        pairs_indices (int tuple list): A list of tuples, where each tuple
            contains the indices of a base pair.
        loop_indices (int list): A list of loop indices.
    """
    # Check that the secondary structure only contains valid characters.
    for c in secondary_structure:
        if c not in NAConstants.open_symbols and \
           c not in NAConstants.close_symbols and \
           c not in NAConstants.loop_symbols:
            raise ValueError(f"Invalid character in secondary structure: {c}")
    
    # Check that the number of open and close symbols are equal.
    num_opens = len([c for c in secondary_structure if c in NAConstants.open_symbols])
    num_closes = len([c for c in secondary_structure if c in NAConstants.close_symbols])
    if num_opens != num_closes:
        raise ValueError(f"Number of open ({num_opens}) and close ({num_closes}) symbols must be equal.")

    pairs_indices = []
    loop_indices = []
    open_symbol_stacks = {open_symbol: [] for open_symbol in NAConstants.open_symbols}
    for i, c in enumerate(secondary_structure):
        # If the symbol is an open symbol, record the index.
        if c in NAConstants.open_symbols:
            open_symbol_stacks[c].append(i)
        # If the symbol is a close symbol, pop the last corresponding open
        # symbol index and record the pair.
        elif c in NAConstants.close_symbols:
            # Get the corresponding open symbol.
            open_symbol = NAConstants.close_to_open[c]

            # Check that there is a corresponding open symbol.
            if len(open_symbol_stacks[open_symbol]) == 0:
                raise ValueError(f"No matching open symbol for close symbol at index {i}.")
            
            # Get the index of the last corresponding open symbol.
            open_index = open_symbol_stacks[open_symbol].pop()

            # Record the pair.
            close_index = i
            pairs_indices.append((open_index, close_index))
        # If the symbol is a loop symbol, record the index.
        elif c in NAConstants.loop_symbols:
            loop_indices.append(i)
        else:
            raise ValueError(f"Invalid character in secondary structure: {c}")
    
    # Check that all open symbols have been closed.
    for open_symbol, open_indices in open_symbol_stacks.items():
        if len(open_indices) > 0:
            raise ValueError(f"No matching close symbol ({NAConstants.open_to_close[open_symbol]}) for open symbol ({open_symbol}) at indices {open_indices}.")

    return pairs_indices, loop_indices

def calculate_secondary_structure_stats(reference_secondary_structure, 
                                        subject_secondary_structure):
    """
    Given a reference secondary structure and a subject secondary structure, 
    calculates the F1 score for the base pairs and loops of the subject.

    Args:
        reference_secondary_structure (str): The reference secondary structure.
        subject_secondary_structure (str): The secondary structure.

    Returns:
        result (dict): A dictionary containing:
            f1_score_pairs (float): The F1 score for the base pairs.
            f1_score_loops (float): The F1 score for the loops.
    """
    # Check that the subject secondary structure and reference secondary
    # structure have the same length.
    if len(subject_secondary_structure) != len(reference_secondary_structure):
        raise ValueError(f"Length of subject secondary structure ({len(subject_secondary_structure)}) must match length of reference secondary structure ({len(reference_secondary_structure)}).")

    # Calculate the base pairs and loops from the secondary structure strings.
    # Also, this function will check the validity of the secondary structures.
    subject_pairs_indices, subject_loop_indices = calculate_base_pairs_and_loops_from_secondary_structure(subject_secondary_structure)
    reference_pairs_indices, reference_loop_indices = calculate_base_pairs_and_loops_from_secondary_structure(reference_secondary_structure)

    # Convert the indices to sets.
    subject_pairs_indices = set(subject_pairs_indices)
    subject_loop_indices = set(subject_loop_indices)

    reference_pairs_indices = set(reference_pairs_indices)
    reference_loop_indices = set(reference_loop_indices)

    # Calculate the number of true positives, false positives, and false 
    # negatives for pairs.
    TP_pairs = len(subject_pairs_indices.intersection(reference_pairs_indices))
    FP_pairs = len(subject_pairs_indices - reference_pairs_indices)
    FN_pairs = len(reference_pairs_indices - subject_pairs_indices)

    # Calculate precision and recall for pairs.
    if TP_pairs + FP_pairs == 0:
        precision_pairs = 0
    else:
        precision_pairs = TP_pairs / (TP_pairs + FP_pairs)
    
    if TP_pairs + FN_pairs == 0:
        recall_pairs = 0
    else:
        recall_pairs = TP_pairs / (TP_pairs + FN_pairs)

    # Calculate F1 score for pairs.
    if precision_pairs + recall_pairs == 0:
        f1_score_pairs = 0
    else:
        f1_score_pairs = 2 * (precision_pairs * recall_pairs) / (precision_pairs + recall_pairs)

    # Calculate the number of true positives, false positives, and false
    # negatives for loops.
    TP_loops = len(subject_loop_indices.intersection(reference_loop_indices))
    FP_loops = len(subject_loop_indices - reference_loop_indices)
    FN_loops = len(reference_loop_indices - subject_loop_indices)

    # Calculate precision and recall for loops.
    if TP_loops + FP_loops == 0:
        precision_loops = 0
    else:
        precision_loops = TP_loops / (TP_loops + FP_loops)
    
    if TP_loops + FN_loops == 0:
        recall_loops = 0
    else:
        recall_loops = TP_loops / (TP_loops + FN_loops)
    
    # Calculate F1 score for loops.
    if precision_loops + recall_loops == 0:
        f1_score_loops = 0
    else:
        f1_score_loops = 2 * (precision_loops * recall_loops) / (precision_loops + recall_loops)
    
    result = {
        "f1_score_pairs": f1_score_pairs,
        "f1_score_loops": f1_score_loops
    }

    return result

def calculate_reactivity_profile_score(reference_secondary_structure,
                                       subject_reactivity_profile):
    """
    Given a reference secondary structure and a subject reactivity profile,
    calculates the EternaFold Classic Score, Crossed Pair Quality Score, and
    OpenKnot score.

    Args:
        reference_secondary_structure (str): The reference secondary structure.
        subject_reactivity_profile (np.ndarray): The reactivity profile.
    
    Returns:
        result (dict): A dictionary containing:
            eternafold_class_score (float): The EternaFold Classic Score.
            crossed_pair_quality_score (float): The Crossed Pair Quality Score.
            openknot_score (float): The OpenKnot score.
    """
    # Setup ARNIE.
    sys.path.append("/projects/ml/afavor/ribonanzanet/")
    with tempfile.NamedTemporaryFile(mode = "wt", suffix = ".txt") as f:
        # Setup the ARNIE config file.
        f.write("linearpartition: . \nTMP: /tmp")
        f.flush()
        arnie_config_path = f.name
        os.environ["ARNIEFILE"] = arnie_config_path
        
        # Import the scoring module from OpenKnotScorePipeline.
        sys.path.append("/projects/ml/afavor/ribonanzanet/kaggle/OpenKnotScorePipeline/openknotscore")
        import scoring

    # Check that the subject reactivity profile and reference secondary 
    # structure have the same length.
    if len(subject_reactivity_profile) != len(reference_secondary_structure):
        raise ValueError(f"Length of subject reactivity profile ({len(subject_reactivity_profile)}) must match length of reference secondary structure ({len(reference_secondary_structure)}).")

    # Check the validity of the reference secondary structure.
    check_secondary_structure_validity(reference_secondary_structure)

    # Convert the reactivity profile to a list.
    subject_reactivity_profile = list(subject_reactivity_profile)

    # Calculate the Eterna Classic Score and Crossed Pair Quality Score.
    eternafold_class_score = \
        scoring.calculateEternaClassicScore(reference_secondary_structure, 
                                            subject_reactivity_profile, 
                                            0, 
                                            0)
    crossed_pair_quality_score = \
        scoring.calculateCrossedPairQualityScore(reference_secondary_structure,
                                                 subject_reactivity_profile,
                                                 0,
                                                 0)[1]

    # Calculate the OpenKnot score.
    openknot_score = (0.5 * eternafold_class_score + 0.5 * crossed_pair_quality_score) / 100

    result = {
        "eternafold_class_score": eternafold_class_score,
        "crossed_pair_quality_score": crossed_pair_quality_score,
        "openknot_score": openknot_score
    }

    return result

################################################################################
# Structure Comparison
################################################################################
def run_us_align(reference_structure_path,
                 subject_structure_path,
                 mol = "RNA",
                 mm = 0,
                 ter = 2,
                 atom = "auto",
                 het = 0,
                 us_align_path = "/projects/ml/afavor/alignment/USalign"):
    """
    Given a reference structure path and a subject structure path, aligns the
    subject structure to the reference structure using US-Align, and calculates
    the root mean square deviation (RMSD) and TM-score between the aligned
    structures.

    Args:
        reference_structure_path (str): The path to the reference structure to
            align to. Reference structure will remain fixed.
        subject_structure_path (str): The path to the structure to align. This 
            structure will be superimposed onto the reference structure.
        mol (str): Type of molecule(s) to align.
            Options:
                "auto": align both protein and nucleic acids.
                "prot": only align proteins in a structure.
                "RNA": (default) only align RNA and DNA in a structure.
        mm (int): Multimeric alignment option.
            Options:
                0: (default) alignment of two monomeric structures.
                1: alignment of two multi-chain oligomeric structures.
                2: alignment of individual chains to an oligomeric structure.
                3: alignment of circularly permuted structure.
                4: alignment of multiple monomeric chains into a consensus alignment.
                5: fully non-sequential (fNS) alignment.
                6: semi-non-sequential (sNS) alignment.
                To use -mm 1 or -mm 2, '-ter' option must be 0 or 1.
        ter (int): Number of chains to align.
            Options:
                3: only align the first chain, or the first segment of the
                    first chain as marked by the 'TER' string in PDB file.
                2: (default) only align the first chain.
                1: align all chains of the first model (recommended for aligning
                    asymmetric units).
                0: align all chains from all models (recommended for aligning
                    biological assemblies, i.e. biounits).
        atom (str): 4-character atom name used to represent a residue. This is
            the atom that will be used to align the structures.
            Options:
                "auto": (default) " C3'" for RNA/DNA and " CA " for proteins.
                four-character atom name: e.g. " C3'" for RNA/DNA and " CA " 
                    for proteins. Note, if mol is set to "auto", atom must also
                    be set to "auto". This is because it is not possible to
                    specify atoms for both protein and nucleic acids. This will
                    result in only the corresponding molecule being aligned.
        het (int): Whether to align residues marked as 'HETATM' in addition to
            'ATOM  '.
            Options:
                0: (default) only align 'ATOM  ' residues.
                1: align both 'ATOM  ' and 'HETATM' residues.
                2: align both 'ATOM  ' and MSE residues.
        us_align_path (str): The path to the US-Align executable.

    Returns:
        result (dict): A dictionary containing:
            rmsd (float): The root mean square deviation (RMSD) between the 
                aligned structures.
            tm_score (float): The TM-score between the aligned structures,
                normalized by the length of the reference structure.
    """
    # Check that the mol and atom options agree.
    if mol == "auto" and atom != "auto":
        raise ValueError("If mol is set to 'auto', atom must also be set to 'auto'.")

    # Convert the structure paths to absolute paths.
    subject_structure_path = os.path.abspath(subject_structure_path)
    reference_structure_path = os.path.abspath(reference_structure_path)

    # Check that the structure paths exist.
    if not os.path.exists(subject_structure_path):
        raise ValueError(f"Structure file not found: {subject_structure_path}")
    if not os.path.exists(reference_structure_path):
        raise ValueError(f"Structure file not found: {reference_structure_path}")

    # Create a temporary file for the US-Align output.
    us_align_output_file = tempfile.NamedTemporaryFile(mode = "wt")

    # Run US-Align.
    try:
        subprocess.run(
            [
                str(us_align_path),
                "-mol",
                str(mol),
                "-mm",
                str(mm),
                "-ter",
                str(ter), 
                "-atom",
                str(atom),
                "-het",
                str(het),
                str(subject_structure_path),
                str(reference_structure_path)
            ],
            check = True,
            stdout = us_align_output_file,
            stderr = subprocess.DEVNULL
        )

        us_align_output_text = read_text_file(us_align_output_file.name)

        # Extract the TM-score and RMSD from the US-Align output.
        rmsd = None
        tm_score = None
        for line in us_align_output_text.split("\n"):
            if line.startswith("Aligned length="):
                rmsd = float(line.split("RMSD=")[1].split(",")[0].strip())
            elif line.startswith("TM-score=") and "normalized by length of Structure_2" in line:
                tm_score = float(line.split("TM-score=")[1].split("(normalized by length of Structure_2")[0].strip())
        
        us_align_output_file.close()
        
        if rmsd is None or tm_score is None:
            raise ValueError("Failed to extract RMSD and TM-score from US-Align output.")
        
        result = {
            "rmsd": rmsd,
            "tm_score": tm_score
        }

        return result
    except (subprocess.CalledProcessError, ValueError) as e:
        us_align_output_file.close()
        raise e

################################################################################
# PPM Comparison
################################################################################
def load_ppms(ppm_paths_str, randomize_experimental_ppms):
    """
    Given a list of lists of ppm paths, with each sublist representing
    multiple experimental alternatives for each ppm, randomly sample one
    ppm path from each sublist, load the ppms from these paths, and
    append their base pairing ppms.

    Arguments:
        ppm_paths_str (str): a string representing a list of lists, where 
            each sublist contains strings of ppm paths. Each sublist 
            represents experimental alternatives for the same ppm.
        randomize_equivalent_ppms (bool): if True, randomly select a random 
            ppm from each sublist of experimental ppms. If False, take the
            first.
    
    Returns:
        ppms ((np.float64 np.ndarray, str) List): a list of ppms 
            (L x 4 arrays) containing the probabilities of DA, DC, DG, DT 
            for DNA and A, C, G, U for RNA, and an associated string
            indicating the ppm type ("dna" or "rna").
        ppm_paths_chosen (str List): a list of the corresponding ppm paths
            that were chosen.

    """
    # Parse the ppm_paths_list_of_lists_str into a list of lists.
    ppm_paths = ast.literal_eval(ppm_paths_str)

    ppms = []
    ppm_paths_chosen = []
    for experimental_ppm_paths_sublist in ppm_paths:
        # Randomly select one of the expermental ppms for each sublist.
        if randomize_experimental_ppms:          
            ppm_path = np.random.choice(experimental_ppm_paths_sublist)
        else:
            ppm_path = experimental_ppm_paths_sublist[0]
        
        # Save the chosen path.
        ppm_paths_chosen.append(ppm_path)

        # Read the ppm, represented as a csv.
        ppm_df = pd.read_csv(ppm_path)

        # Create the ppm.
        if "T" in ppm_df.columns:
            ppm = np.stack((np.array(ppm_df["A"], dtype = np.float64),
                            np.array(ppm_df["C"], dtype = np.float64),
                            np.array(ppm_df["G"], dtype = np.float64),
                            np.array(ppm_df["T"], dtype = np.float64)),
                        axis = -1)
            ppm_type = "dna"
        elif "U" in ppm_df.columns:
            ppm = np.stack((np.array(ppm_df["A"], dtype = np.float64),
                            np.array(ppm_df["C"], dtype = np.float64),
                            np.array(ppm_df["G"], dtype = np.float64),
                            np.array(ppm_df["U"], dtype = np.float64)),
                        axis = -1)
            ppm_type = "rna"
        else:
            raise Exception(f"PPM at {ppm_path} is not valid.")

        # Compute the base pairing ppm.
        bp_ppm = np.copy(np.flip(np.flip(ppm, axis = 1), axis = 0))

        ppms.append((ppm, ppm_type))
        ppms.append((bp_ppm, ppm_type))

    return ppms, ppm_paths_chosen
    
def calculate_information_content(ppm, eps = 1e-10):
    """
    Calculate the per-position information content of a position probability
    matrix (PPM).

    Arguments:
        ppm (np.float64 np.ndarray): an L x 4 array of representing the PPM.
        eps (float, optional): a small epsilon, to be added to each 
            probability to prevent taking a logarithm of 0.
    
    Returns:
        per_position_ic (np.float64 np.ndarray): the per-position
            information content of the ppm.
    """
    # Check the shape of the ppm.
    assert(ppm.shape[-1] == 4)

    # Add epsilon to each column of probabilities and normalize.
    ppm_plus_eps = ppm + eps
    ppm_eps_norm = ppm_plus_eps / np.sum(ppm_plus_eps, axis = -1)[:, None]

    # Compute the per-position information content.
    per_position_ic = np.sum(np.log(ppm_eps_norm) / np.log(0.25), axis = -1)

    return per_position_ic

def calculate_pearson_correlation_coeffcient(ppm, S_one_hot):
    """
    Calculate the per-position pearson correlation coefficient between the
    ppm and the (one-hot) nucleic acid sequence.

    Arguments:
        ppm (np.float64 np.ndarray): an L x 4 array of representing the PPM
        S_one_hot (np.float64 np.ndarray): an L x 4 array representing the
            one-hot nucleic acid sequence.
    
    Returns:
        per_position_pcc (np.float64 np.ndarray): the per-position pearson
            correlation coefficient between the ppm and one-hot sequence.
            Note, if the ppm is the uniform ppm, returns 0 by default.
    """
    # Check the shape of the ppm and the S_one_hot.
    assert(ppm.shape[-1] == 4)
    assert(S_one_hot.shape[-1] == 4)
    
    # Compute the per-position mean of the ppm and S_one_hot
    ppm_bar = np.mean(ppm, axis = -1)
    S_one_hot_bar = np.mean(S_one_hot, axis = -1)

    # Calculate the per-position pearson correlation coefficient between
    # the ppm and the one-hot sequence.
    numerator = np.sum((ppm - ppm_bar[:, None]) * (S_one_hot - S_one_hot_bar[:, None]), axis = -1)
    denominator = np.sqrt(np.sum((ppm - ppm_bar[:, None]) ** 2, axis = -1) * np.sum((S_one_hot - S_one_hot_bar[:, None]) ** 2, axis = -1))
    
    # Since S_one_hot is a one-hot vectors, the only way that the
    # denominator is 0 is if the ppm is uniform. If this is the case,
    # return 0 by default. This is okay since this is used as an alignment
    # scoring mechanism.
    denominator_non_zero_mask = denominator != 0
    per_position_pcc = np.zeros_like(numerator)
    np.place(per_position_pcc, denominator_non_zero_mask, 
                numerator[denominator_non_zero_mask] / denominator[denominator_non_zero_mask])

    return per_position_pcc

def calculate_alignment_score(ppm, S_one_hot):
    """
    Calculate the per-position information content-weighted pearson
    correlation coefficient between the ppm and the one-hot nucleic acid 
    sequence, and return the sum.

    Arguments:
        ppm (np.float64 np.ndarray): an L x 4 array of representing the PPM.
        S_one_hot (np.float64 np.ndarray): an L x 4 array representing the
            one-hot nucleic acid sequence.
    
    Returns:
        ic_weighted_pcc_sum (float): the sum of the per-position information 
            content-weighted pearson correlation coefficient between 
            the ppm and the one-hot DNA sequence.
    """
    # Check the shape of the ppm and the S_one_hot.
    assert(ppm.shape[-1] == 4)
    assert(S_one_hot.shape[-1] == 4)

    # Calculate the per-position information content-weighted pearson
    # correlation coefficient.
    per_position_ic = calculate_information_content(ppm)
    per_position_pcc = calculate_pearson_correlation_coeffcient(ppm, S_one_hot)
    per_position_ic_weighted_pcc = per_position_pcc * (0.5 * per_position_ic)

    # Calculate the sum of the per_position_ic_weighted_pcc.
    ic_weighted_pcc_sum = np.sum(per_position_ic_weighted_pcc)

    return ic_weighted_pcc_sum

def weighted_align(ppm, S_one_hot_na, S_non_x_mask):
    """
    Given a ppm and a sequence, find the maximum information 
    content-weighted pearson correlation coefficient alignment(s).

    Arguments:
        ppm (np.float64 np.ndarray): an L x 4 array of representing the PPM.
        S_one_hot_na (np.float64 np.ndarray): an L x 4 array representing 
            the nucleic acid column subset of the sequence one-hot vector.
        S_non_x_mask (bool np.ndarray): an L length mask that is True if
            the corresponding column in S_one_hot_na has a 1, and False
            otherwise.
    
    Returns:
        max_score (float): the score of the best alignment.
        opt_ppm_starts (int List): the starting index in the ppm for the
            maximum score alignments.
        opt_S_starts (int List): the starting index in the sequence for the
            maximum score alignments.
        opt_overlap_lens (int List): the overlap lengths of the maximum
            score alignments.
    """
    # Check the shape of the ppm and the S_one_hot_na.
    assert(ppm.shape[-1] == 4)
    assert(S_one_hot_na.shape[-1] == 4)

    max_score = -1 * np.inf
    opt_ppm_starts = [0]
    opt_S_starts = [0]
    opt_overlap_lens = [0]

    ppm_len = ppm.shape[0]
    S_len = S_one_hot_na.shape[0]
    
    # Check all possible start positions in the ppm and sequence, as well
    # as all possible overlap lengths.
    for ppm_start in range(ppm_len):
        for overlap_len in range(ppm_len - ppm_start + 1):
            for S_start in range(S_len - overlap_len + 1):
                # Get the chunks of the ppm and sequence.
                ppm_chunk = ppm[ppm_start : ppm_start + overlap_len]
                S_one_hot_chunk = S_one_hot_na[S_start : S_start + overlap_len]
                S_non_x_mask_chunk = S_non_x_mask[S_start : S_start + overlap_len]

                # If the overlap or the number of non-DX tokens in the 
                # sequence chunk are less than the defined minimum overlap
                # length, continue.
                if overlap_len < NAConstants.min_overlap_length or \
                    np.count_nonzero(S_non_x_mask_chunk) < NAConstants.min_overlap_length:
                    continue

                # Remove any DX parts of the sequence and the corresponding
                # part of the ppm.
                ppm_chunk = ppm_chunk[S_non_x_mask_chunk]
                S_one_hot_chunk = S_one_hot_chunk[S_non_x_mask_chunk]

                score = calculate_alignment_score(ppm_chunk, S_one_hot_chunk)

                if score > max_score:
                    max_score = score
                    opt_ppm_starts = [ppm_start]
                    opt_S_starts = [S_start]
                    opt_overlap_lens = [overlap_len]
                elif score == max_score:
                    opt_ppm_starts.append(ppm_start)
                    opt_S_starts.append(S_start)
                    opt_overlap_lens.append(overlap_len)

    return max_score, opt_ppm_starts, opt_S_starts, opt_overlap_lens

def align_ppms(ppms, S, chain_labels, protein_mask, dna_mask, rna_mask):
    """
    Given a list of ppms (L x 4), a polymer sequence, the polymer chain
    labels, and a mask of which specifies which residues are protein
    residues, align each ppm against every chain of S, recording the best
    alignments in an aligned ppm.

    Arguments:
        ppms ((np.float64 np.ndarray, str) List): a list of ppms 
            (L x 4 arrays) containing the probabilities of DA, DC, DG, DT 
            for DNA and A, C, G, U for RNA, and an associated string
            indicating the ppm type ("dna" or "rna").
        S (np.int32 np.ndarray): an L length array representing the
            sequence tokens.
        chain_labels (np.int32 np.ndarray): an L length array containing
            the chain label of every residue.
        protein_mask (np.int32 np.ndarray): an L length array that is
            1 if the residue is a protein residue and 0 otherwise.
        dna_mask (np.int32 np.ndarray): an L length array that is
            1 if the residue is a DNA residue and 0 otherwise.
        rna_mask (np.int32 np.ndarray): an L length array that is
            1 if the residue is a RNA residue and 0 otherwise.
    
    Returns:
        aligned_ppm (np.float64 np.ndarray): an 
            (L x len(self.restype_to_int) array, containing the ppm
            information aligned to the sequence. 
        ppm_mask (np.int32 np.ndarray): an L length array that is 1 if
            ppm information has been aligned for the residue and 0 
            otherwise.
        alignment_score (float): the alignment score of the overall
            alignment.
    """
    aligned_ppm = np.zeros((S.shape[0], len(NAConstants.na_mpnn_restype_to_int)), dtype = np.float64)
    ppm_mask = np.zeros_like(S, dtype = np.int32)

    # Create a one-hot vector from the sequence.
    S_len = S.shape[0]
    S_one_hot = np.zeros((S_len, len(NAConstants.na_mpnn_restype_to_int)), dtype = np.float64)
    S_one_hot[np.arange(S_len), S] = 1
    
    # For each ppm, align against every chain, and record the best
    # alignments.
    unique_chains = np.unique(chain_labels)
    for (ppm, ppm_type) in ppms:
        max_score = -1 * np.inf
        opt_ppm_starts = []
        opt_S_starts = []
        opt_overlap_lens = []

        # Extract the appropriate nucleic acid columns from the one-hot 
        # sequence.
        if ppm_type == "dna":
            na_restype_ints_to_compare = [NAConstants.na_mpnn_restype_to_int["DA"], 
                                          NAConstants.na_mpnn_restype_to_int["DC"], 
                                          NAConstants.na_mpnn_restype_to_int["DG"],
                                          NAConstants.na_mpnn_restype_to_int["DT"]]
        elif ppm_type == "rna":
            na_restype_ints_to_compare = [NAConstants.na_mpnn_restype_to_int["A"], 
                                          NAConstants.na_mpnn_restype_to_int["C"], 
                                          NAConstants.na_mpnn_restype_to_int["G"],
                                          NAConstants.na_mpnn_restype_to_int["U"]]
        S_one_hot_na = S_one_hot[:, na_restype_ints_to_compare]  
        
        # Create a mask where the nucleic acid columns are not all zero.
        S_non_x_mask = np.sum(S_one_hot_na, axis = -1) > 0

        for chain_label in unique_chains:
            # Get the sequence and start index for every chain, as well as
            # the NA one-hot and non-unknown mask subset for the chain.
            chain_indices = np.where(chain_labels == chain_label)
            chain_S_start_idx = chain_indices[0][0]
            chain_S_one_hot_na = S_one_hot_na[chain_indices]
            chain_S_non_x_mask = S_non_x_mask[chain_indices]

            # Exclude protein chains.
            if protein_mask[chain_S_start_idx] == 1:
                continue
            # Make sure the PPM type matches the chain type.
            elif (dna_mask[chain_S_start_idx] == 1) and (ppm_type == "rna"):
                continue
            elif (rna_mask[chain_S_start_idx] == 1) and (ppm_type == "dna"):
                continue
            
            # Align the ppm against the chain sequence.
            (chain_max_score, 
                chain_opt_ppm_starts, 
                chain_opt_S_starts, 
                chain_opt_overlap_lens) = weighted_align(ppm, chain_S_one_hot_na, chain_S_non_x_mask)
            
            # Adjust the sequence start positions to represent the overall
            # index, not just the index within the chain.
            chain_opt_S_starts = \
                list(map(lambda relative_index: relative_index + chain_S_start_idx, 
                            chain_opt_S_starts))
            
            # If the alignment score is greater than previous scores,
            # set the optimal alignments to the current ones. If the
            # alignment score is the same, extend the optimal alignments
            # with the current ones.
            if chain_max_score > max_score:
                max_score = chain_max_score
                opt_ppm_starts = copy.copy(chain_opt_ppm_starts)
                opt_S_starts = copy.copy(chain_opt_S_starts)
                opt_overlap_lens = copy.copy(chain_opt_overlap_lens)
            elif chain_max_score == max_score:
                opt_ppm_starts.extend(chain_opt_ppm_starts)
                opt_S_starts.extend(chain_opt_S_starts)
                opt_overlap_lens.extend(chain_opt_overlap_lens)

        # If the max score is greater than the minimum score, write the
        # optimal alignments into the aligned ppm.
        if max_score > (-1 * np.inf):
            for (opt_ppm_start, opt_S_start, opt_overlap_len) in \
                zip(opt_ppm_starts, opt_S_starts, opt_overlap_lens):
                for shared_idx in range(opt_overlap_len):
                    ppm_idx = opt_ppm_start + shared_idx
                    S_idx = opt_S_start + shared_idx
                    
                    # If a ppm column has already been written to the
                    # specified sequence index in the aligned ppm, choose
                    # the ppm column with the higher score (if the DNA is
                    # not DX at that position), and higher information
                    # content otherwise.
                    if ppm_mask[S_idx] == 0:
                        aligned_ppm[S_idx, na_restype_ints_to_compare] = ppm[ppm_idx]
                        ppm_mask[S_idx] = 1
                    elif ppm_mask[S_idx] == 1:
                        if S_non_x_mask[S_idx]:
                            ppm_score = calculate_alignment_score(ppm[ppm_idx][None, :], S_one_hot_na[S_idx][None, :])
                            aligned_ppm_score = calculate_alignment_score(aligned_ppm[S_idx, na_restype_ints_to_compare][None, :], S_one_hot_na[S_idx][None, :])
                            if ppm_score > aligned_ppm_score:
                                aligned_ppm[S_idx, na_restype_ints_to_compare] = ppm[ppm_idx]
                        else:
                            ppm_col_ic = calculate_information_content(ppm[ppm_idx][None, :])
                            aligned_ppm_col_ic = calculate_information_content(aligned_ppm[S_idx, na_restype_ints_to_compare][None, :])
                            if ppm_col_ic > aligned_ppm_col_ic:
                                aligned_ppm[S_idx, na_restype_ints_to_compare] = ppm[ppm_idx]

    # Calculate the final alignment score for DNA.
    S_non_dx_mask = (S != NAConstants.na_mpnn_restype_to_int["DX"])
    position_mask_dna = np.logical_and(np.logical_and(ppm_mask == 1, S_non_dx_mask == 1), 
                                       dna_mask == 1)
    if np.count_nonzero(position_mask_dna) == 0:
        alignment_score_dna = np.nan
        aligned_dna_length = 0
    else:
        # Subset based on position.
        aligned_ppm_dna = aligned_ppm[position_mask_dna]
        S_one_hot_dna = S_one_hot[position_mask_dna]

        # Subset based on residue type.
        residue_type_mask_dna = \
            np.array([restype in NAConstants.deep_pbs_restypes for restype in NAConstants.na_mpnn_restypes])
        aligned_ppm_dna = aligned_ppm_dna[:, residue_type_mask_dna]
        S_one_hot_dna = S_one_hot_dna[:, residue_type_mask_dna]

        # Calculate the alignment score.
        alignment_score_dna = calculate_alignment_score(
            aligned_ppm_dna,
            S_one_hot_dna
        )
        aligned_dna_length = np.count_nonzero(position_mask_dna)
    
    # Calculate the final alignment score for RNA.
    S_non_rx_mask = (S != NAConstants.na_mpnn_restype_to_int["RX"])
    position_mask_rna = np.logical_and(np.logical_and(ppm_mask == 1, S_non_rx_mask == 1), 
                                       rna_mask == 1)
    if np.count_nonzero(position_mask_rna) == 0:
        alignment_score_rna = np.nan
        aligned_rna_length = 0
    else:
        # Subset based on position.
        aligned_ppm_rna = aligned_ppm[position_mask_rna]
        S_one_hot_rna = S_one_hot[position_mask_rna]

        # Subset based on residue type.
        residue_type_mask_rna = \
            np.array([restype in NAConstants.rna_restypes for restype in NAConstants.na_mpnn_restypes])
        aligned_ppm_rna = aligned_ppm_rna[:, residue_type_mask_rna]
        S_one_hot_rna = S_one_hot_rna[:, residue_type_mask_rna]

        # Calculate the alignment score.
        alignment_score_rna = calculate_alignment_score(
            aligned_ppm_rna,
            S_one_hot_rna
        )
        aligned_rna_length = np.count_nonzero(position_mask_rna)

    return aligned_ppm, ppm_mask, alignment_score_dna, aligned_dna_length, \
        alignment_score_rna, aligned_rna_length

def calculate_ppm_mean_absolute_error(reference_ppm, subject_ppm):
    """
    Given a reference PPM and a subject PPM, calculates the mean absolute error
    between the two PPM matrices.

    Args:
        reference_ppm (np.ndarray): The reference PPM.
        subject_ppm (np.ndarray): The subject PPM.
    
    Returns:
        result (dict): A dictionary containing:
            mean_absolute_error (float): The mean absolute error between the two 
                PPM matrices.
    """
    # Check that the two PPMs have the same shape.
    if subject_ppm.shape != reference_ppm.shape:
        raise ValueError(f"The subject PPM shape ({subject_ppm.shape}) must match the reference PPM shape ({reference_ppm.shape}).")

    # Calculate the mean absolute error.
    L = subject_ppm.shape[0]
    mean_absolute_error = (1 / L) * np.sum(np.abs(subject_ppm - reference_ppm))

    result = {
        "mean_absolute_error": mean_absolute_error
    }
    return result

def calculate_ppm_root_mean_squared_error(reference_ppm, subject_ppm):
    """
    Given a reference PPM and a subject PPM, calculates the root mean squared
    error between the two PPM matrices.

    Args:
        reference_ppm (np.ndarray): The reference PPM.
        subject_ppm (np.ndarray): The subject PPM.
    
    Returns:
        result (dict): A dictionary containing:
            root_mean_squared_error (float): The root mean squared error between 
                the two PPM matrices.
    """
    # Check that the two PPMs have the same shape.
    if subject_ppm.shape != reference_ppm.shape:
        raise ValueError(f"The subject PPM shape ({subject_ppm.shape}) must match the reference PPM shape ({reference_ppm.shape}).")
    
    # Calculate the root mean squared error.
    L = subject_ppm.shape[0]
    root_mean_squared_error = np.sqrt((1 / L) * np.sum((subject_ppm - reference_ppm) ** 2))

    result = {
        "root_mean_squared_error": root_mean_squared_error
    }

    return result

def calculate_ppm_cross_entropy(reference_ppm, subject_ppm):
    """
    Given a reference PPM and a subject PPM, calculates the cross entropy
    between the two PPM matrices.

    Args:
        reference_ppm (np.ndarray): The reference PPM.
        subject_ppm (np.ndarray): The subject PPM.
    
    Returns:
        result (dict): A dictionary containing:
            cross_entropy (float): The cross entropy between the two PPM 
                matrices.
    """
    # Check that the two PPMs have the same shape.
    if subject_ppm.shape != reference_ppm.shape:
        raise ValueError(f"The subject PPM shape ({subject_ppm.shape}) must match the reference PPM shape ({reference_ppm.shape}).")
    
    # Calculate the cross entropy.
    L = subject_ppm.shape[0]
    cross_entropy = - (1 / L) * np.sum(reference_ppm * np.log(subject_ppm))

    result = {
        "cross_entropy": cross_entropy
    }

    return result

def calculate_specificity_score():
    pass

################################################################################
# Sequence design
################################################################################
def run_na_mpnn_sequence(structure_path, 
                         output_directory = None,
                         batch_size = 1,
                         number_of_batches = 1,
                         temperature = 0.1,
                         omit_AA = "",
                         design_na_only = 0,
                         load_residues_with_missing_atoms = 0,
                         output_pdbs = 0,
                         catch_failed_inferences = 1,
                         na_mpnn_apptainer_path = "/software/containers/mlfold.sif",
                         na_mpnn_path = "/home/akubaney/projects/fused_mpnn/run.py",
                         na_mpnn_model_path = None):
    """
    Given a structure path, runs the NA-MPNN sequence design algorithm to
    generate sequences for the structure. The output is a list of dictionaries
    containing the design ID, name, design sequence, and tool-reported sequence
    recovery.

    Args:
        structure_path (str): The path to the structure file.
        output_directory (str): The path to the output directory. If not
            specified, a temporary directory will be created.
        batch_size (int): The batch size for the NA-MPNN algorithm.
        number_of_batches (int): The number of batches to run.
        temperature (float): The temperature for the NA-MPNN algorithm.
        omit_AA (str): The amino acids to omit from the design.
        design_na_only (int): Whether to design only nucleic acids.
        load_residues_with_missing_atoms (int): Whether to load residues with
            missing atoms.
        output_pdbs (int): Whether to output PDB files.
        catch_failed_inferences (int): Whether to catch failed inferences.
        na_mpnn_apptainer_path (str): The path to the NA-MPNN apptainer.
        na_mpnn_path (str): The path to the NA-MPNN run file.
        na_mpnn_model_path (str): The path to the NA-MPNN model file.
    
    Returns:
        design_data (dict list): A list of dictionaries containing:
            input_structure_name (str): The name of the input structure.
            input_structure_path (str): The path to the input structure.
            design_id (str): The design ID.
            name (str): The name of the design.
            design_sequence (str): The design sequence.
            tool_reported_sequence_recovery (float): The tool-reported sequence
                recovery.
            design_method (str): The design method used.
            model_weights_path (str): The path to the model weights used.
    """
    # Convert the structure path to an absolute path.
    structure_path = os.path.abspath(structure_path)

    # Check that the structure path exists.
    if not os.path.exists(structure_path):
        raise ValueError(f"Structure file not found: {structure_path}")

    # If the output directory is not specified, create a temporary directory.
    # The temporary directory will be automatically cleaned up when the script
    # exits.
    if output_directory is None:
        tmp_directory = tempfile.TemporaryDirectory()
        output_directory = tmp_directory.name
    else:
        output_directory = os.path.abspath(output_directory)
    
    # Compute the output directory for the sequences.
    seqs_output_directory = os.path.join(output_directory, "seqs")
    
    # Compute the name of the structure.
    structure_name = os.path.splitext(os.path.basename(structure_path))[0]

    # Run the NA-MPNN sequence design algorithm.
    try:
        subprocess.run(
            [
                "apptainer",
                "exec",
                na_mpnn_apptainer_path,
                "python",
                na_mpnn_path,
                "--model_type",
                str("na_mpnn"),
                "--checkpoint_na_mpnn",
                str(na_mpnn_model_path),
                "--pdb_path",
                str(structure_path),
                "--out_folder",
                str(output_directory),
                "--number_of_batches",
                str(number_of_batches),
                "--batch_size",
                str(batch_size),
                "--temperature",
                str(temperature),
                "--omit_AA",
                str(omit_AA),
                "--design_na_only",
                str(design_na_only),
                "--load_residues_with_missing_atoms",
                str(load_residues_with_missing_atoms),
                "--output_pdbs",
                str(output_pdbs),
                "--catch_failed_inferences",
                str(catch_failed_inferences)                
            ],
            check = True,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        # Check that the output fasta file exists.
        fasta_path = os.path.join(seqs_output_directory, f"{structure_name}.fa")
        if not os.path.exists(fasta_path):
            raise ValueError(f"Output fasta file not found: {fasta_path}")

        # Read the output fasta file.
        fasta_entries = read_fasta_file(fasta_path)

        # Skip the first entry of the fasta, which contains the parent sequence.
        fasta_entries = fasta_entries[1:]

        design_data = []
        for fasta_header, fasta_sequence in fasta_entries:
            fasta_header = fasta_header.strip()
            fasta_header_metadata = fasta_header.split(", ")

            metadata_dict = dict()
            for metadata in fasta_header_metadata[1:]:
                metadata = metadata.strip()
                metadata_name, metadata_value = metadata.split("=")
                metadata_dict[metadata_name] = metadata_value

            design_dict = {
                "input_structure_name": structure_name,
                "input_structure_path": structure_path,
                "design_id": metadata_dict["id"],
                "name": f"{structure_name}_{metadata_dict['id']}",
                "design_sequence": fasta_sequence,
                "tool_reported_sequence_recovery": float(metadata_dict["seq_rec"]),
                "design_method": "na_mpnn",
                "model_weights_path": na_mpnn_model_path
            }

            design_data.append(design_dict)

        # Clean up the temporary directory if it was created.
        if output_directory is None:
            tmp_directory.cleanup()

        return design_data
    except (subprocess.CalledProcessError, ValueError) as e:
        # Clean up the temporary directory if it was created.
        if output_directory is None:
            tmp_directory.cleanup()
        raise e
    
def run_grnade(structure_path,
               output_directory = None,
               n_samples = 1,
               temperature = 0.1,
               grnade_apptainer_path = "/software/containers/users/akubaney/grnade.sif",
               grnade_path = "/home/akubaney/software/gRNAde/gRNAde.py"):
    """
    Given a structure path, runs the gRNAde sequence design algorithm to
    generate sequences for the structure. The output is a list of dictionaries
    containing the design ID, name, design sequence, and tool-reported sequence
    recovery.

    Args:
        structure_path (str): The path to the structure file.
        output_directory (str): The path to the output directory. If not
            specified, a temporary directory will be created.
        n_samples (int): The number of samples to generate.
        temperature (float): The temperature for the gRNAde algorithm.
        grnade_apptainer_path (str): The path to the gRNAde apptainer.
        grnade_path (str): The path to the gRNAde run file.
    
    Returns:
        design_data (dict list): A list of dictionaries containing:
            input_structure_name (str): The name of the input structure.
            input_structure_path (str): The path to the input structure.
            design_id (str): The design ID.
            name (str): The name of the design.
            design_sequence (str): The design sequence.
            tool_reported_sequence_recovery (float): The tool-reported sequence
                recovery.
            design_method (str): The design method used.
            model_weights_path (str): The path to the model weights used.
    """
    # Convert the structure path to an absolute path.
    structure_path = os.path.abspath(structure_path)

    # Check that the structure path exists.
    if not os.path.exists(structure_path):
        raise ValueError(f"Structure file not found: {structure_path}")
    
    # If the output directory is not specified, create a temporary directory.
    # The temporary directory will be automatically cleaned up when the script
    # exits.
    if output_directory is None:
        tmp_directory = tempfile.TemporaryDirectory()
        output_directory = tmp_directory.name
    else:
        output_directory = os.path.abspath(output_directory)
    
    # Compute the output directory for the sequences.
    seqs_output_directory = os.path.join(output_directory, "seqs")

    # Create the output directory if it does not exist.
    os.makedirs(seqs_output_directory, exist_ok = True)

    # Compute the name of the structure.
    structure_name = os.path.splitext(os.path.basename(structure_path))[0]

    # Run the gRNAde sequence design algorithm.
    try:
        subprocess.run(
            [
                "apptainer",
                "exec",
                grnade_apptainer_path,
                "python",
                grnade_path,
                "--pdb_filepath",
                str(structure_path),
                "--output_filepath",
                os.path.join(seqs_output_directory, f"{structure_name}.fa"),
                "--split",
                "das",
                "--max_num_conformers",
                str(1),
                "--n_samples",
                str(n_samples),
                "--temperature",
                str(temperature)
            ],
            check = True,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        # Check that the output fasta file exists.
        fasta_path = os.path.join(seqs_output_directory, f"{structure_name}.fa")
        if not os.path.exists(fasta_path):
            raise ValueError(f"Output fasta file not found: {fasta_path}")

        # Read the output fasta file.
        fasta_entries = read_fasta_file(fasta_path)

        # Skip the first entry of the fasta, which contains the parent sequence.
        fasta_entries = fasta_entries[1:]

        design_data = []
        for fasta_header, fasta_sequence in fasta_entries:
            fasta_header = fasta_header.strip()
            fasta_header_metadata = fasta_header.split(", ")

            metadata_dict = dict()
            for metadata in fasta_header_metadata:
                metadata = metadata.strip()
                metadata_name, metadata_value = metadata.split("=")
                metadata_dict[metadata_name] = metadata_value
            
            design_dict = {
                "input_structure_name": structure_name,
                "input_structure_path": structure_path,
                "design_id": metadata_dict["sample"],
                "name": f"{structure_name}_{metadata_dict['sample']}",
                "design_sequence": fasta_sequence.replace("\n", ""),
                "tool_reported_sequence_recovery": float(metadata_dict["recovery"]),
                "design_method": "grnade",
                "model_weights_path": ""
            }
        
            design_data.append(design_dict)
        
        # Clean up the temporary directory if it was created.
        if output_directory is None:
            tmp_directory.cleanup()

        return design_data
    except (subprocess.CalledProcessError, ValueError) as e:
        # Clean up the temporary directory if it was created.
        if output_directory is None:
            tmp_directory.cleanup()
        raise e

def run_rhodesign(structure_path,
                  output_directory = None,
                  n_samples = 1,
                  temperature = 0.1,
                  rhodesign_apptainer_path = "/software/containers/users/akubaney/rhodesign.sif",
                  rhodesign_path = "/home/akubaney/software/RhoDesign/src/inference_without2d.py"):
    """
    Given a structure path, runs the RhoDesign sequence design algorithm to
    generate sequences for the structure. The output is a list of dictionaries
    containing the design ID, name, design sequence, and tool-reported sequence
    recovery.

    Args:
        structure_path (str): The path to the structure file.
        output_directory (str): The path to the output directory. If not
            specified, a temporary directory will be created.
        n_samples (int): The number of samples to generate.
        temperature (float): The temperature for the RhoDesign algorithm.
        rhodesign_apptainer_path (str): The path to the RhoDesign apptainer.
        rhodesign_path (str): The path to the RhoDesign run file.
    
    Returns:
        design_data (dict list): A list of dictionaries containing:
            input_structure_name (str): The name of the input structure.
            input_structure_path (str): The path to the input structure.
            design_id (str): The design ID.
            name (str): The name of the design.
            design_sequence (str): The design sequence.
            tool_reported_sequence_recovery (float): The tool-reported sequence
                recovery.
            design_method (str): The design method used.
            model_weights_path (str): The path to the model weights used.
    """
    # Convert the structure path to an absolute path.
    structure_path = os.path.abspath(structure_path)

    # Check that the structure path exists.
    if not os.path.exists(structure_path):
        raise ValueError(f"Structure file not found: {structure_path}")
    
    # If the output directory is not specified, create a temporary directory.
    # The temporary directory will be automatically cleaned up when the script
    # exits.
    if output_directory is None:
        tmp_directory = tempfile.TemporaryDirectory()
        output_directory = tmp_directory.name
    else:
        output_directory = os.path.abspath(output_directory)
    
    # Compute the output directory for the sequences.
    seqs_output_directory = os.path.join(output_directory, "seqs")

    # Create the output directory if it does not exist.
    os.makedirs(seqs_output_directory, exist_ok = True)

    # Compute the name of the structure.
    structure_name = os.path.splitext(os.path.basename(structure_path))[0]

    # Run the RhoDesign sequence design algorithm.
    try:
        fasta_entries = []
        design_data = []
        for i in range(n_samples):
            # Create a temporary directory for the output.
            output_directory_i = tempfile.TemporaryDirectory()

            # Create a temporary file for the standard output.
            output_file_i = tempfile.NamedTemporaryFile(mode = "wt", suffix = ".txt")

            subprocess.run(
                [
                    "apptainer",
                    "exec",
                    rhodesign_apptainer_path,
                    "python",
                    rhodesign_path,
                    "-pdb",
                    str(structure_path),
                    "-save",
                    str(output_directory_i.name),
                    "-temp",
                    str(temperature)
                ],
                check = True,
                stdout = output_file_i,
                stderr = subprocess.DEVNULL
            )

            # Do not keep the output saved by RhoDesign.
            output_directory_i.cleanup()

            # Read and close the ith output file.
            output_text = read_text_file(output_file_i.name)
            output_file_i.close()

            # Extract the sequence and sequence recovery from the output.
            for line in output_text.split("\n"):
                if line.startswith("sequence: "):
                    sequence = line.split(": ")[1].strip()
                elif line.startswith("recovery rate: "):
                    tool_reported_sequence_recovery = line.split(": ")[1].strip()
            
            # Add an entry to the fasta.
            fasta_entries.append(
                (f">{structure_name}, id={i}, seq_rec={tool_reported_sequence_recovery}", 
                 sequence)
            )

            # Create a dictionary for the design data.
            design_dict = {
                "input_structure_name": structure_name,
                "input_structure_path": structure_path,
                "design_id": str(i),
                "name": f"{structure_name}_{i}",
                "design_sequence": sequence,
                "tool_reported_sequence_recovery": float(tool_reported_sequence_recovery),
                "design_method": "rhodesign",
                "model_weights_path": ""
            }

            design_data.append(design_dict)

        # Write the fasta entries to a file.
        fasta_path = os.path.join(seqs_output_directory, f"{structure_name}.fa")
        write_fasta_file(fasta_path, fasta_entries)

        # Clean up the temporary directory if it was created.
        if output_directory is None:
            tmp_directory.cleanup()

        return design_data
    except (subprocess.CalledProcessError, ValueError) as e:
        # Clean up the temporary directory if it was created.
        if output_directory is None:
            tmp_directory.cleanup()
        
        # Clean up the output file and output directory for the ith sample.
        output_directory_i.cleanup()
        output_file_i.close()

        raise e

################################################################################
# Combined Functionality
################################################################################
def design_nucleic_acid_sequence(structure_path,
                                 overall_output_directory,
                                 num_samples,
                                 temperature,
                                 method = "na_mpnn",
                                 na_mpnn_model_path = None):
    """
    Given a structure path, an overall output directory, the number of samples,
    the temperature, and the sequence design method, runs the specified
    sequence design method to generate sequences for the structure. A JSON is
    created for each design, containing the design ID, name, design sequence,
    and tool-reported sequence recovery.

    Args:
        structure_path (str): The path to the structure file.
        overall_output_directory (str): The path to the overall output directory.
        num_samples (int): The number of samples to generate.
        temperature (float): The temperature for the sequence design algorithm.
        method (str): The sequence design method to use. Options are "na_mpnn",
            "grnade", and "rhodesign". Default is "na_mpnn".
        na_mpnn_model_path (str): The path to the NA-MPNN model file. Required
            if method is "na_mpnn".
    
    Side Effects:
        Creates an output directory for the structure, copies the structure to
            the output directory, creates a subdirectory for the design JSON
            files, and saves a JSON file for each design containing the design
            ID, name, design sequence, and tool-reported sequence recovery.
    """
    # Convert the structure path and overall output directory to absolute paths.
    structure_path = os.path.abspath(structure_path)
    overall_output_directory = os.path.abspath(overall_output_directory)

    if temperature is None:
        temperature = 0.1
    
    if na_mpnn_model_path is None:
        na_mpnn_model_path = "/home/akubaney/projects/na_mpnn/models/design_model/s_19137.pt"

    # Check that the structure path exists.
    if not os.path.exists(structure_path):
        raise ValueError(f"Structure file not found: {structure_path}")
    
    # Create the overall output directory if it does not exist.
    os.makedirs(overall_output_directory, exist_ok = True)

    # Get the basename without the ".gz" extension.
    if structure_path.endswith(".gz"):
        structure_basename = os.path.splitext(os.path.basename(structure_path))[0]
    else:
        structure_basename = os.path.basename(structure_path)
    # Extract the name of the structure (without the extension).
    if structure_basename.endswith(".pdb") or structure_basename.endswith(".cif"):
        structure_name = os.path.splitext(structure_basename)[0]
    else:
        raise ValueError(f"Invalid structure file extension: {structure_basename}")

    # Create the specific output directory for the structure. If the directory
    # already exists, remove it and create a new one.
    output_directory = os.path.join(overall_output_directory, structure_name)
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # Copy the structure to the output directory. If it is a gzipped file,
    # decompress it first.
    copy_structure_path = os.path.join(output_directory, structure_basename)
    if structure_path.endswith(".gz"):
        with gzip.open(structure_path, "rb") as f_in:
            with open(copy_structure_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy(structure_path, copy_structure_path)

    # Save the original and new structure paths.
    original_structure_path = structure_path
    structure_path = copy_structure_path

    # Design JSON output directory.
    design_json_output_directory = os.path.join(output_directory, "design_json")
    os.makedirs(design_json_output_directory)

    if method == "na_mpnn":
        # Run NA-MPNN sequence design.
        design_data = run_na_mpnn_sequence(
            structure_path,
            output_directory = output_directory,
            batch_size = num_samples,
            number_of_batches = 1,
            temperature = temperature,
            omit_AA = "ARNDCQEGHILKMFPSTWYVXbdhuy",
            design_na_only = 1,
            load_residues_with_missing_atoms = 0,
            output_pdbs = 0,
            catch_failed_inferences = 1,
            na_mpnn_model_path = na_mpnn_model_path
        )
    elif method == "grnade":
        # Run gRNAde sequence design.
        design_data = run_grnade(
            structure_path,
            output_directory = output_directory,
            n_samples = num_samples,
            temperature = temperature
        )
    elif method == "rhodesign":
        # Run RhoDesign sequence design.
        design_data = run_rhodesign(
            structure_path,
            output_directory = output_directory,
            n_samples = num_samples,
            temperature = temperature
        )
    else:
        raise ValueError(f"Invalid sequence design method: {method}")

    # Write the design data to a JSON file.
    for design_dict in design_data:
        design_dict["original_input_structure_path"] = original_structure_path
        design_json_path = os.path.join(
            design_json_output_directory,
            f"{design_dict['name']}.json"
        )
        write_json_file(design_json_path, design_dict)            

def process_reference_monomer_rna(reference_structure_path, 
                                  overall_output_directory):
    """
    Given a reference structure path and an overall output directory,
    processes the reference structure, extracts its sequence and secondary
    structure with DSSR, and saves the results to a JSON file.

    Args:
        reference_structure_path (str): The path to the reference structure.
        overall_output_directory (str): The path to the overall output 
            directory.
    
    Side Effects:
        Creates an output directory for the reference structure, copies the 
            reference structure to the output directory, and saves a JSON file 
            with the results of the predictions.
    """
    # Convert the structure path and overall output directory to absolute paths.
    reference_structure_path = os.path.abspath(reference_structure_path)
    overall_output_directory = os.path.abspath(overall_output_directory)

    # Check that the reference structure path.
    if not os.path.exists(reference_structure_path):
        raise ValueError(f"Reference structure file not found: {reference_structure_path}")
    
    # Create the output directory if it does not exist.
    os.makedirs(overall_output_directory, exist_ok = True)
    
    # Get the basename without the ".gz" extension.
    if reference_structure_path.endswith(".gz"):
        reference_structure_basename = os.path.splitext(os.path.basename(reference_structure_path))[0]
    else:
        reference_structure_basename = os.path.basename(reference_structure_path)
    # Extract the name of the structure (without the extension).
    if reference_structure_basename.endswith(".pdb") or reference_structure_basename.endswith(".cif"):
        structure_name = os.path.splitext(reference_structure_basename)[0]
    else:
        raise ValueError(f"Invalid structure file extension: {reference_structure_basename}")

    # Create the specific output directory for the structure. If the directory
    # already exists, remove it and create a new one.
    output_directory = os.path.join(overall_output_directory, structure_name)
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # Copy the reference structure to the output directory. If it is a gzipped
    # file, decompress it first.
    copy_reference_structure_path = os.path.join(output_directory, reference_structure_basename)
    if reference_structure_path.endswith(".gz"):
        with gzip.open(reference_structure_path, "rb") as f_in:
            with open(copy_reference_structure_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy(reference_structure_path, copy_reference_structure_path)
    
    # Save the original and new structure paths.
    original_reference_structure_path = reference_structure_path
    reference_structure_path = copy_reference_structure_path

    # Create the output directory for the reference json results.
    reference_json_output_directory = os.path.join(output_directory, "reference_json")
    os.makedirs(reference_json_output_directory)

    # Run dssr.
    dssr_output = run_dssr(reference_structure_path)
    
    # Standardize the dssr sequence.
    dssr_output["sequence"] = \
        standardize_rna_sequence(dssr_output["sequence"], 
                                 method = "dssr")
    
    # Check that sequence is valid.
    check_rna_sequence_validity(dssr_output["sequence"],
                                unknown_residue_allowed = True,
                                chain_breaks_allowed = False)
    
    # Standardize the dssr secondary structure.
    dssr_output["secondary_structure"] = \
        standardize_secondary_structure(dssr_output["secondary_structure"], 
                                        method = "dssr")

    output_dict = {
        "name": structure_name,
        "original_reference_structure_path": original_reference_structure_path,
        "reference_structure_path": reference_structure_path,
        "dssr": dssr_output,
    }

    # Save the output dictionary to a JSON file.
    output_json_path = os.path.join(reference_json_output_directory, 
                                    f"{structure_name}.json")
    write_json_file(output_json_path, output_dict)

def process_design_monomer_rna(subject_path, 
                               overall_output_directory):
    """
    Given a design path and an overall output directory, processes the design
    by extracting its sequence and secondary structure with DSSR, predicting
    its secondary structure with EternaFold, predicting its secondary
    structure and reactivity profile with RiboNanzaNet, and predicting its
    structure with AlphaFold3. The results are saved to a JSON file.
    
    Args:
        subject_path (str): The path to the design JSON file.
        overall_output_directory (str): The path to the overall output 
            directory.
    
    Side Effects:
        Creates an output directory for the design, copies the design fasta 
            file to the output directory, and saves a JSON file with the 
            results of the predictions.
    """
    # Convert the subject path and overall output directory to absolute paths.
    subject_path = os.path.abspath(subject_path)
    overall_output_directory = os.path.abspath(overall_output_directory)

    # Check that the subject path exists.
    if not os.path.exists(subject_path):
        raise ValueError(f"Design fasta file not found: {subject_path}")
    
    # Create the output directory if it does not exist.
    os.makedirs(overall_output_directory, exist_ok = True)
    
    # Read the subject JSON file.
    design_json = read_json_file(subject_path)

    # Get the name of the design.
    design_name = design_json["name"]

    # Create the specific output directory for the design. If the directory
    # already exists, remove it and create a new one.
    output_directory = os.path.join(overall_output_directory, design_name)
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # Create the output directory for the processed design json results.
    processed_design_json_output_directory = os.path.join(output_directory, "processed_design_json")
    os.makedirs(processed_design_json_output_directory)

    # Get the design sequence.
    design_sequence = design_json["design_sequence"]
    design_method = design_json["design_method"]

    # Standardize the design sequence.
    design_sequence = standardize_rna_sequence(design_sequence, 
                                               method = design_method)
    
    # Check that sequence is valid.
    check_rna_sequence_validity(design_sequence,
                                unknown_residue_allowed = False,
                                chain_breaks_allowed = False)

    # Predict the secondary structure of the design sequence with EternaFold.
    eternafold_result = run_eternafold(design_sequence)

    # Predict the secondary structure and reactivity profile of the design
    # sequence with RiboNanzaNet.
    ribonanza_net_secondary_structure_result = \
        run_ribonanza_net_secondary_structure(design_sequence)
    ribonanza_net_reactivity_profile_result = \
        run_ribonanza_net_reactivity_profile(design_sequence)
    
    # # Predict the structure of the design sequence with AlphaFold3.
    alphafold3_result = run_alphafold3(
        name = design_name, 
        sequences_and_polytypes = [(design_sequence, "rna")], 
        output_dir = output_directory, 
        num_diffusion_samples = 5, 
        num_seeds = 1,
        run_data_pipeline = False,
        buckets = "1"
    )

    # Create the output dictionary.
    output_dict = {
        "name": design_name,
        "sequence": design_sequence,
        "design_input_path": subject_path,
        "eternafold": eternafold_result,
        "ribonanza_net_secondary_structure": ribonanza_net_secondary_structure_result,
        "ribonanza_net_reactivity_profile": ribonanza_net_reactivity_profile_result,
        "alphafold3": alphafold3_result
    }

    # Save the output dictionary to a JSON file.
    output_json_path = os.path.join(processed_design_json_output_directory, 
                                    f"{design_name}.json")
    write_json_file(output_json_path, output_dict)

def score_design_monomer_rna(reference_path, subject_path, overall_output_directory):
    """
    Given a reference path and a subject path, scores the design by comparing
    the reference and subject sequences, secondary structures, reactivity
    profiles, and structures.

    Args:
        reference_path (str): The path to the reference output json.
        subject_path (str): The path to the subject output json.
        overall_output_directory (str): The path to the overall output 
            directory.
    
    Side Effects:
        Creates an output directory for the subject and saves a JSON file
            with the results of the scoring.
    """
    import biotite
    import biotite.structure.io

    # Convert the reference path and subject path to absolute paths.
    reference_path = os.path.abspath(reference_path)
    subject_path = os.path.abspath(subject_path)

    # Check that the reference path exists.
    if not os.path.exists(reference_path):
        raise ValueError(f"Reference structure file not found: {reference_path}")
    
    # Check that the subject path exists.
    if not os.path.exists(subject_path):
        raise ValueError(f"Subject structure file not found: {subject_path}")
    
    # Create the output directory if it does not exist.
    os.makedirs(overall_output_directory, exist_ok = True)

    # Load the reference output.
    reference_output = read_json_file(reference_path)

    # Load the subject output.
    subject_output = read_json_file(subject_path)

    # Make the output directory for the subject if it does not exist. If the
    # directory already exists, remove it and create a new one.
    output_directory = os.path.join(overall_output_directory,
                                    subject_output["name"])
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # Load the C1' atoms from the reference and subject structures.
    subject_atom_array = biotite.structure.io.load_structure(subject_output["alphafold3"]["predicted_structure_path"])
    reference_atom_array = biotite.structure.io.load_structure(reference_output["reference_structure_path"])
    
    reference_atom_array = reference_atom_array[reference_atom_array.atom_name == "C1'"]
    subject_atom_array = subject_atom_array[subject_atom_array.atom_name == "C1'"]

    # Handle the case where the subject sequence is shorter than the reference
    # sequence. This can happen if residues at the end get chopped off.
    subject_sequence_length = len(subject_output["sequence"])
    reference_sequence_length = len(reference_output["dssr"]["sequence"])
    if subject_sequence_length == reference_sequence_length:
        best_start_idx = None
        best_end_idx = None
    elif subject_sequence_length < reference_sequence_length:
        # Perform an rmsd calculation to determine the best overlap.
        best_rmsd = None
        best_start_idx = None
        for possible_start_idx in range(reference_sequence_length - subject_sequence_length + 1):
            reference_start_idx = possible_start_idx
            reference_end_idx = reference_start_idx + subject_sequence_length

            # Subset the reference atom array.
            reference_atom_subarray = reference_atom_array[
                reference_start_idx:reference_end_idx
            ]

            # Superimpose the reference and subject atom arrays.
            superimposed, _ = biotite.structure.superimpose(
                reference_atom_subarray,
                subject_atom_array
            )

            # Calculate the RMSD.
            c1_prime_rmsd = biotite.structure.rmsd(
                reference_atom_subarray,
                superimposed
            )

            if best_rmsd is None or c1_prime_rmsd < best_rmsd:
                best_rmsd = c1_prime_rmsd
                best_start_idx = possible_start_idx
        
        best_end_idx = best_start_idx + subject_sequence_length

        # Subset the sequence, and atom array to the best overlap.
        reference_output["dssr"]["sequence"] = \
            reference_output["dssr"]["sequence"][best_start_idx:best_end_idx]
        reference_atom_array = reference_atom_array[best_start_idx:best_end_idx]

        # The secondary structure needs to be modified in an appropriate way
        # (any base pairs to the removed residues need to be removed).
        base_pair_indices, _ = calculate_base_pairs_and_loops_from_secondary_structure(
            reference_output["dssr"]["secondary_structure"]
        )
        updated_secondary_structure = reference_output["dssr"]["secondary_structure"]
        for (i, j) in base_pair_indices:
            if i < best_start_idx or j < best_start_idx or \
               i >= best_end_idx or j >= best_end_idx:
                
                # Turn i and j indices into loops.
                updated_secondary_structure = \
                    updated_secondary_structure[:i] + \
                    NAConstants.loop_symbols[0] + \
                    updated_secondary_structure[i + 1:]
                updated_secondary_structure = \
                    updated_secondary_structure[:j] + \
                    NAConstants.loop_symbols[0] + \
                    updated_secondary_structure[j + 1:]
        
        # Now trim the updated secondary structure.
        reference_output["dssr"]["secondary_structure"] = \
            updated_secondary_structure[best_start_idx:best_end_idx]
    else:
        raise ValueError("Subject sequence is longer than reference sequence.")
    
    # Compare the sequences.
    sequence_recovery_result = calculate_sequence_recovery(
        reference_output["dssr"]["sequence"],
        subject_output["sequence"]
    )

    # Compare the reference secondary structure to the eternafold predicted
    # secondary structure.
    eternafold_secondary_structure_result = \
        calculate_secondary_structure_stats(
            reference_output["dssr"]["secondary_structure"],
            subject_output["eternafold"]["predicted_secondary_structure"]
        )
    
    # Compare the reference secondary structure to the ribonanza net
    # predicted secondary structures.
    ribonanza_net_secondary_structure_result = dict()
    for predicted_secondary_structure in subject_output["ribonanza_net_secondary_structure"]["predicted_secondary_structures"]:
        individual_result = \
            calculate_secondary_structure_stats(
                reference_output["dssr"]["secondary_structure"],
                predicted_secondary_structure
            )
        
        # Append the results for each ribonanza net predicted secondary
        # structure to the ribonanza net secondary structure result.
        for metric_name, metric_value in individual_result.items():
            if metric_name not in ribonanza_net_secondary_structure_result:
                ribonanza_net_secondary_structure_result[metric_name] = []
            ribonanza_net_secondary_structure_result[metric_name].append(metric_value)
    
    # Calculate the mean of the ribonanza net secondary structure results.
    for metric_name, metric_values in list(ribonanza_net_secondary_structure_result.items()):
        ribonanza_net_secondary_structure_result[f"mean_{metric_name}"] = \
            np.mean(metric_values)
        
    # Compare the reference secondary structure to the ribonanza net
    # predicted reactivity profiles.
    ribonanza_net_reactivity_profile_result = dict()
    for predicted_reactivity_profile in subject_output["ribonanza_net_reactivity_profile"]["predicted_2A3_reactivity_profiles"]:
        individual_result = \
            calculate_reactivity_profile_score(
                reference_output["dssr"]["secondary_structure"],
                predicted_reactivity_profile
            )
        
        # Append the results for each ribonanza net predicted reactivity
        # profile to the ribonanza net reactivity profile result.
        for metric_name, metric_value in individual_result.items():
            if metric_name not in ribonanza_net_reactivity_profile_result:
                ribonanza_net_reactivity_profile_result[metric_name] = []
            ribonanza_net_reactivity_profile_result[metric_name].append(metric_value)
        
    # Calculate the mean of the ribonanza net reactivity profile results.
    for metric_name, metric_values in list(ribonanza_net_reactivity_profile_result.items()):
        ribonanza_net_reactivity_profile_result[f"mean_{metric_name}"] = \
            np.mean(metric_values)

    # Check that the reference and subject structures contain the same number
    # of C1' atoms.
    if reference_atom_array.shape[0] != subject_atom_array.shape[0]:
        raise ValueError("Reference and subject structures must contain the same number of C1' atoms.")
    
    superimposed, _ = biotite.structure.superimpose(
        reference_atom_array,
        subject_atom_array
    )
    c1_prime_rmsd = biotite.structure.rmsd(
        reference_atom_array,
        superimposed
    )

    c1_prime_lddt = biotite.structure.lddt(
        reference_atom_array,
        subject_atom_array
    )

    c1_prime_gddt = biotite.structure.lddt(
        reference_atom_array,
        subject_atom_array,
        inclusion_radius = 10000,
        distance_bins = (1.0, 2.0, 4.0, 8.0)
    )

    # Create the output dictionary.
    output_dict = {
        "reference_name": reference_output["name"],
        "reference_path": reference_path,
        "reference_sequence_length": reference_sequence_length,
        "subject_name": subject_output["name"],
        "subject_path": subject_path,
        "subject_sequence_length": subject_sequence_length,
        "best_start_idx": best_start_idx,
        "best_end_idx": best_end_idx,
        "sequence_recovery": sequence_recovery_result["sequence_recovery"],
        "eternafold_f1_score_pairs": eternafold_secondary_structure_result["f1_score_pairs"],
        "eternafold_f1_score_loops": eternafold_secondary_structure_result["f1_score_loops"],
        "ribonanza_net_f1_score_pairs": ribonanza_net_secondary_structure_result["mean_f1_score_pairs"],
        "ribonanza_net_f1_score_loops": ribonanza_net_secondary_structure_result["mean_f1_score_loops"],
        "ribonanza_net_eternafold_class_score": ribonanza_net_reactivity_profile_result["mean_eternafold_class_score"],
        "ribonanza_net_crossed_pair_quality_score": ribonanza_net_reactivity_profile_result["mean_crossed_pair_quality_score"],
        "ribonanza_net_openknot_score": ribonanza_net_reactivity_profile_result["mean_openknot_score"],
        "alphafold3_c1_prime_rmsd": float(c1_prime_rmsd),
        "alphafold3_c1_prime_lddt": c1_prime_lddt,
        "alphafold3_c1_prime_gddt": c1_prime_gddt,
        "alphafold3_ptm": subject_output["alphafold3"]["ptm"],
        "alphafold3_pae": subject_output["alphafold3"]["pae"],
        "alphafold3_plddt": subject_output["alphafold3"]["plddt"]
    }

    # Save the output dictionary to a JSON file.
    output_json_path = os.path.join(output_directory, 
                                    f"{subject_output['name']}.json")
    write_json_file(output_json_path, output_dict)

def predict_nucleic_acid_ppm(structure_path,
                             overall_output_directory,
                             num_samples,
                             temperature,
                             method = "na_mpnn",
                             na_mpnn_model_path = None):
    """
    Given a structure path, an overall output directory, the number of samples,
    the temperature, and the specificity prediction method, runs the specified
    specificity prediction method to generate specificity predictions for the
    structure. A JSON is created for each prediction, containing the predicted
    specificity, true sequence, and polymer type.

    Args:
        structure_path (str): The path to the structure file.
        overall_output_directory (str): The path to the overall output 
            directory.
        num_samples (int): The number of samples to generate.
        temperature (float): The temperature for the specificity prediction
            algorithm.
        ppm_polymer_type (str): The polymer type for the specificity prediction.
            Options are "dna" and "rna". Default is "dna".
        method (str): The specificity prediction method to use. Options are
            "na_mpnn" and "deeppbs". Default is "na_mpnn".
        na_mpnn_model_path (str): The path to the NA-MPNN model file. Required
            if method is "na_mpnn".
    
    Side Effects:
        Creates an output directory for the structure, copies the structure to
            the output directory, creates a subdirectory for the specificity
            JSON files, and saves a JSON file for each prediction containing
            the predicted specificity, true sequence, and polymer type.
    """
    if num_samples is None:
        num_samples = 30
    
    if temperature is None:
        temperature = 0.6

    if na_mpnn_model_path is None:
        na_mpnn_model_path = "/home/akubaney/projects/na_mpnn/models/specificity_model/s_70114.pt"
    
    # Convert the structure path and overall output directory to absolute paths.
    structure_path = os.path.abspath(structure_path)
    overall_output_directory = os.path.abspath(overall_output_directory)

    # Check that the structure path exists.
    if not os.path.exists(structure_path):
        raise ValueError(f"Invalid structure path: {structure_path}")
    
    # Create the overall output directory if it does not exist.
    os.makedirs(overall_output_directory, exist_ok = True)

    # Get the basename without the ".gz" extension.
    if structure_path.endswith(".gz"):
        structure_basename = os.path.splitext(os.path.basename(structure_path))[0]
    else:
        structure_basename = os.path.basename(structure_path)
    # Extract the name of the structure (without the extension).
    if structure_basename.endswith(".pdb") or structure_basename.endswith(".cif"):
        structure_name = os.path.splitext(structure_basename)[0]
    else:
        raise ValueError(f"Invalid structure file extension: {structure_basename}")

    # Create the specific output directory for the structure. If the directory
    # already exists, remove it and create a new one.
    output_directory = os.path.join(overall_output_directory, structure_name)
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # Copy the structure to the output directory. If it is a gzipped file,
    # decompress it first.
    copy_structure_path = os.path.join(output_directory, structure_basename)
    if structure_path.endswith(".gz"):
        with gzip.open(structure_path, "rb") as f_in:
            with open(copy_structure_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy(structure_path, copy_structure_path)

    # Save the original and new structure paths.
    original_structure_path = structure_path
    structure_path = copy_structure_path

    # Specificity JSON output directory.
    specificity_json_output_directory = os.path.join(output_directory, "specificity_json")
    os.makedirs(specificity_json_output_directory)

    if method == "na_mpnn":
        # Run NA-MPNN specificity prediction.
        specificity_data = run_na_mpnn_specificity(
            structure_path,
            output_directory = output_directory,
            batch_size = num_samples,
            number_of_batches = 1,
            temperature = temperature,
            omit_AA = "ARNDCQEGHILKMFPSTWYVXbdhuy",
            design_na_only = 1,
            load_residues_with_missing_atoms = 0,
            output_pdbs = 0,
            output_sequences = 0,
            output_specificity = 1,
            catch_failed_inferences = 1,
            na_mpnn_model_path = na_mpnn_model_path,
        )
    elif method == "deeppbs":
        # Run DeepPBS specificity prediction.
        specificity_data = run_deeppbs(
            structure_path,
            output_directory = output_directory
        )
    else:
        raise ValueError(f"Invalid specificity prediction method: {method}")
    
    # Add the original path to the specificity data.
    specificity_data["original_input_structure_path"] = original_structure_path

    # Convert all numpy arrays to lists; in preparation for saving to JSON.
    for k, v in specificity_data.items():
        if isinstance(v, np.ndarray):
            specificity_data[k] = v.tolist()
    
    # Save the specificity data to a JSON file.
    specificity_json_path = os.path.join(specificity_json_output_directory, f"{structure_name}.json")
    write_json_file(specificity_json_path, specificity_data)

def score_specificity_prediction(reference_ppms_list_str, 
                                 subject_path, 
                                 overall_output_directory):
    """
    Given a reference PPM list string and a subject path, scores the
    specificity prediction by comparing the reference and subject PPMs.

    Args:
        reference_ppms_list_str (str): The reference PPM list string.
        subject_path (str): The path to the subject JSON file.
        overall_output_directory (str): The path to the overall output 
            directory.
    
    Side Effects:
        Creates an output directory for the subject and saves a JSON file
            with the results of the scoring.
    """
    # Convert the subject path and overall output directory to absolute paths.
    subject_path = os.path.abspath(subject_path)
    overall_output_directory = os.path.abspath(overall_output_directory)
    
    # Check that the subject path exists.
    if not os.path.exists(subject_path):
        raise ValueError(f"Predicted PPM file not found: {subject_path}")

    # Create the overall output directory if it does not exist.
    os.makedirs(overall_output_directory, exist_ok = True)
    
    # Load the subject dictionary.
    subject_output = read_json_file(subject_path)

    # Make the output directory for the subject if it does not exist. If the
    # directory already exists, remove it and create a new one.
    output_directory = os.path.join(overall_output_directory,
                                    subject_output["name"])
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # Load the reference PPMs, with no randomization of the ppms for 
    # consistent scoring.
    reference_ppms, reference_ppm_paths_chosen = load_ppms(
        reference_ppms_list_str,
        randomize_experimental_ppms = False
    )
    
    # Convert the necessary fields to numpy arrays.
    subject_true_sequence_na_mpnn_format = np.array(
        subject_output["true_sequence_na_mpnn_format"],
        dtype = np.int64
    )
    subject_chain_labels = np.array(
        subject_output["chain_labels"],
        dtype = np.int32
    )
    subject_protein_mask = np.array(
        subject_output["protein_mask"],
        dtype = np.int32
    )
    subject_dna_mask = np.array(
        subject_output["dna_mask"],
        dtype = np.int32
    )
    subject_rna_mask = np.array(
        subject_output["rna_mask"],
        dtype = np.int32
    )

    # Align the reference PPMs to the subject true sequence.
    aligned_ppm, ppm_mask, alignment_score_dna, aligned_dna_length, \
        alignment_score_rna, aligned_rna_length = align_ppms(
        reference_ppms,
        subject_true_sequence_na_mpnn_format,
        subject_chain_labels,
        subject_protein_mask,
        subject_dna_mask,
        subject_rna_mask
    )

    # Load the subject predicted PPMs.
    subject_predicted_ppm_na_mpnn_format = np.array(
        subject_output["predicted_ppm_na_mpnn_format"],
        dtype = np.float64
    )
    subject_mask = np.array(
        subject_output["mask"],
        dtype = np.int32
    )

    # Subset the aligned ppm and subject predicted ppm to where the ppm_mask
    # exists (the ground truth ppm exists) and the subject mask exists (the
    # subject predicted ppm exists). Also, mask the residue type dimension 
    # appropriately. Calculate the specificity metrics for DNA.
    position_mask_dna = np.logical_and(np.logical_and(ppm_mask == 1, subject_mask == 1), 
                                       subject_dna_mask == 1)
    if np.count_nonzero(position_mask_dna) == 0:
        mean_absolute_error_dna = np.nan
        root_mean_squared_error_dna = np.nan
        cross_entropy_dna = np.nan
    else:
        # Subset based on position.
        aligned_ppm_dna = aligned_ppm[position_mask_dna]
        subject_ppm_dna = subject_predicted_ppm_na_mpnn_format[position_mask_dna]

        # Subset based on residue type.
        residue_type_mask_dna = \
            np.array([restype in NAConstants.deep_pbs_restypes for restype in NAConstants.na_mpnn_restypes])
        aligned_ppm_dna = aligned_ppm_dna[:, residue_type_mask_dna]
        subject_ppm_dna = subject_ppm_dna[:, residue_type_mask_dna]

        mean_absolute_error_dna = calculate_ppm_mean_absolute_error(
            aligned_ppm_dna,
            subject_ppm_dna
        )
        root_mean_squared_error_dna = calculate_ppm_root_mean_squared_error(
            aligned_ppm_dna,
            subject_ppm_dna
        )
        cross_entropy_dna = calculate_ppm_cross_entropy(
            aligned_ppm_dna,
            subject_ppm_dna
        )
    
    # Calculate the specificity metrics for rna.
    position_mask_rna = np.logical_and(np.logical_and(ppm_mask == 1, subject_mask == 1),
                                       subject_rna_mask == 1)
    if np.count_nonzero(position_mask_rna) == 0:
        mean_absolute_error_rna = np.nan
        root_mean_squared_error_rna = np.nan
        cross_entropy_rna = np.nan
    else:
        # Subset based on position.
        aligned_ppm_rna = aligned_ppm[position_mask_rna]
        subject_ppm_rna = subject_predicted_ppm_na_mpnn_format[position_mask_rna]

        # Subset based on residue type.
        residue_type_mask_rna = \
            np.array([restype in NAConstants.rna_restypes for restype in NAConstants.na_mpnn_restypes])
        aligned_ppm_rna = aligned_ppm_rna[:, residue_type_mask_rna]
        subject_ppm_rna = subject_ppm_rna[:, residue_type_mask_rna]
        
        mean_absolute_error_rna = calculate_ppm_mean_absolute_error(
            aligned_ppm_rna,
            subject_ppm_rna
        )
        root_mean_squared_error_rna = calculate_ppm_root_mean_squared_error(
            aligned_ppm_rna,
            subject_ppm_rna
        )
        cross_entropy_rna = calculate_ppm_cross_entropy(
            aligned_ppm_rna,
            subject_ppm_rna
        )

    # Create the result dictionary.
    result = {
        "reference_ppms_list_str": reference_ppms_list_str,
        "reference_ppm_paths_chosen": reference_ppm_paths_chosen,
        "subject_name": subject_output["name"],
        "subject_path": subject_path,
        "aligned_ppm": aligned_ppm,
        "ppm_mask": ppm_mask,
        "alignment_score_dna": alignment_score_dna,
        "aligned_dna_length": aligned_dna_length,
        "mean_absolute_error_dna": mean_absolute_error_dna,
        "root_mean_squared_error_dna": root_mean_squared_error_dna,
        "cross_entropy_dna": cross_entropy_dna,
        "alignment_score_rna": alignment_score_rna,
        "aligned_rna_length": aligned_rna_length,
        "mean_absolute_error_rna": mean_absolute_error_rna,
        "root_mean_squared_error_rna": root_mean_squared_error_rna,
        "cross_entropy_rna": cross_entropy_rna,
    }

    # Convert numpy arrays to lists for JSON serialization.
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
    
    # Save the result to a JSON file.
    output_json_path = os.path.join(output_directory,
                                    f"{subject_output['name']}.json")
    write_json_file(output_json_path, result)

def score_design_specificity():
    pass

################################################################################
# Run from Command Line
################################################################################
if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argument_parser.add_argument(
        "--function_name", 
        type = str,
        help = "The name of the function to run."
    )
    argument_parser.add_argument(
        "--structure_path", 
        type = str,
        help = "The path to the structure file."
    )
    argument_parser.add_argument(
        "--overall_output_directory", 
        type = str,
        help = "The path to the overall output directory."
    )
    argument_parser.add_argument(
        "--num_samples", 
        type = int,
        help = "The number of samples to generate.",
        default = None
    )
    argument_parser.add_argument(
        "--temperature", 
        type = float,
        help = "The temperature for the sequence design algorithm.",
        default = None
    )
    argument_parser.add_argument(
        "--method", 
        type = str,
        help = "The method to use."
    )
    argument_parser.add_argument(
        "--na_mpnn_model_path", 
        type = str,
        help = "The path to the NA-MPNN model file.",
        default = None
    )
    argument_parser.add_argument(
        "--reference_structure_path", 
        type = str,
        help = "The path to the reference structure."
    )
    argument_parser.add_argument(
        "--subject_path", 
        type = str,
        help = "The path to the subject data."
    )
    argument_parser.add_argument(
        "--reference_path", 
        type = str,
        help = "The path to the reference data."
    )
    argument_parser.add_argument(
        "--reference_ppms_list_str", 
        type = str,
        help = "The reference PPM list string."
    )

    # Parse the command line arguments.
    args = argument_parser.parse_args()

    if args.function_name == "design_nucleic_acid_sequence":
        design_nucleic_acid_sequence(args.structure_path,
                                     args.overall_output_directory,
                                     args.num_samples,
                                     args.temperature,
                                     method = args.method,
                                     na_mpnn_model_path = args.na_mpnn_model_path)
    elif args.function_name == "process_reference_monomer_rna":
        process_reference_monomer_rna(args.reference_structure_path,
                                      args.overall_output_directory)
    elif args.function_name == "process_design_monomer_rna":
        process_design_monomer_rna(args.subject_path,
                                   args.overall_output_directory)
    elif args.function_name == "score_design_monomer_rna":
        score_design_monomer_rna(args.reference_path,
                                 args.subject_path,
                                 args.overall_output_directory)
    elif args.function_name == "predict_nucleic_acid_ppm":
        predict_nucleic_acid_ppm(args.structure_path,
                                 args.overall_output_directory,
                                 args.num_samples,
                                 args.temperature,
                                 method = args.method,
                                 na_mpnn_model_path = args.na_mpnn_model_path)
    elif args.function_name == "score_specificity_prediction":
        score_specificity_prediction(args.reference_ppms_list_str,
                                     args.subject_path,
                                     args.overall_output_directory)
    else:
        raise ValueError(f"Function {args.function_name} not recognized.")