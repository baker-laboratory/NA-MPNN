################################################################################
# Imports
################################################################################
# Python Standard Libraries
import argparse
import ast
import copy
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


# Third-Party Libraries
import numpy as np
import pandas as pd

import biotite
import biotite.structure

import atomworks
import atomworks.io.utils.io_utils
from atomworks.enums import ChainType
from atomworks.ml.utils.token import get_token_starts

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
        contents = deserialize_json_enums(contents)
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
        json.dump(serialize_json_enums(contents), f, indent = 4)

def serialize_json_enums(contents):
    """
    Recursively converts enum values into JSON-safe tagged dictionaries.

    Args:
        contents (any): The contents to serialize.

    Returns:
        serialized_contents (any): The serialized contents.
    """
    if isinstance(contents, ChainType):
        return {"__type__": "ChainType", "name": contents.name}
    if isinstance(contents, dict):
        return {
            key: serialize_json_enums(value)
            for key, value in contents.items()
        }
    if isinstance(contents, (list, tuple)):
        return [serialize_json_enums(value) for value in contents]
    return contents

def deserialize_json_enums(contents):
    """
    Recursively restores tagged enum values loaded from JSON.

    Args:
        contents (any): The JSON-loaded contents.

    Returns:
        deserialized_contents (any): The deserialized contents.
    """
    if isinstance(contents, dict):
        if (
            contents.get("__type__") == "ChainType" and
            set(contents.keys()) == {"__type__", "name"}
        ):
            return ChainType[contents["name"]]
        return {
            key: deserialize_json_enums(value)
            for key, value in contents.items()
        }
    if isinstance(contents, list):
        return [deserialize_json_enums(value) for value in contents]
    return contents

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
    true_sequence_na_mpnn_format = specificity_data[
        true_sequence_na_mpnn_format_key
    ]
    mask = specificity_data["mask"]
    if ppm_polymer_type == "dna":
        polymer_mask = specificity_data["dna_mask"]
        restype_to_int = NAConstants.deep_pbs_restype_to_int
    elif ppm_polymer_type == "rna":
        if method == "deeppbs":
            raise ValueError(
                "DeepPBS does not support RNA specificity prediction."
            )
        polymer_mask = specificity_data["rna_mask"]
        restype_to_int = NAConstants.rna_restype_to_int
    else:
        raise ValueError(f"Invalid polymer type: {ppm_polymer_type}")
    
    # Subset by mask and polymer mask.
    true_sequence = true_sequence_na_mpnn_format[
        np.logical_and(mask == 1, polymer_mask == 1)
    ]

    # Convert to the (possibly shared token) representation.
    true_sequence = [
        NAConstants.na_mpnn_int_to_restype[restype_int]
        for restype_int in true_sequence
    ]

    # If shared token representation is used and the polymer type is RNA,
    # convert from shared to RNA tokens.
    if NAConstants.na_mpnn_na_shared_tokens and ppm_polymer_type == "rna":
        true_sequence = [
            NAConstants.dna_restype_to_rna_restype[restype]
            for restype in true_sequence 
        ]

    return true_sequence

def load_first_assembly_parsed_and_atom_array(
    structure_path,
    add_missing_atoms = True
):
    """
    Load the first bioassembly if present, otherwise load the asymmetric unit.

    Args:
        structure_path (str or file-like): The path to the structure file to
            load, or a file-like object containing the structure text.
        add_missing_atoms (bool): Whether AtomWorks should add missing atoms
            while parsing. True by default.

    Returns:
        parsed (dict): The parsed structure dictionary.
        atom_array (atomworks AtomArray): The atom array for the first
            bioassembly if present, otherwise the atom array for the asymmetric
            unit.
    """
    structure_suffix = None
    if isinstance(structure_path, (str, os.PathLike)):
        structure_suffix = pathlib.Path(structure_path).suffix.lower()
    elif hasattr(structure_path, "name"):
        structure_suffix = pathlib.Path(structure_path.name).suffix.lower()

    # PDB input is already de-symmetrized for evaluation and may intentionally
    # keep non-canonical residues in polymer chains. AtomWorks' top-level PDB
    # parser rewrites mixed polymer/non-polymer chains into separate chains,
    # so for PDB files we load the raw AtomArray first and then derive the
    # usual annotations with parse_atom_array().
    if structure_suffix == ".pdb":
        raw_atom_array = atomworks.io.utils.io_utils.load_any(
            structure_path,
            model = 1,
            altloc = "first",
            extra_fields = ["b_factor", "occupancy", "charge", "atom_id"]
        )
        try:
            parsed = atomworks.io.parser.parse_atom_array(
                raw_atom_array,
                _cif_file = None,
                add_missing_atoms = add_missing_atoms,
                fix_formal_charges = add_missing_atoms
            )
        except Exception:
            # Handle a particular edge case ligand.
            parsed = atomworks.io.parser.parse_atom_array(
                raw_atom_array,
                _cif_file = None,
                add_missing_atoms = add_missing_atoms,
                fix_formal_charges = add_missing_atoms,
                remove_ccds = ["SPW"]
            )
    else:
        # Load the structure.
        try:
            parsed = atomworks.io.parser.parse(
                structure_path,
                add_missing_atoms = add_missing_atoms,
                fix_formal_charges = add_missing_atoms
            )
        except Exception:
            # Handle a particular edge case ligand.
            parsed = atomworks.io.parser.parse(
                structure_path,
                add_missing_atoms = add_missing_atoms,
                fix_formal_charges = add_missing_atoms,
                remove_ccds = ["SPW"]
            )

    # Use the first bioassembly if it exists, otherwise fall back to the
    # asymmetric unit.
    if ("assemblies" in parsed) and (len(parsed["assemblies"]) > 0):
        first_assembly_id = list(parsed["assemblies"].keys())[0]
        atom_array = parsed["assemblies"][first_assembly_id][0]
    else:
        atom_array = parsed["asym_unit"][0]

    return parsed, atom_array


def load_first_assembly_atom_array(structure_path,
                                   add_missing_atoms = True):
    """
    Load the first bioassembly if present, otherwise load the asymmetric unit.

    Args:
        structure_path (str or file-like): The path to the structure file to
            load, or a file-like object containing the structure text.
        add_missing_atoms (bool): Whether AtomWorks should add missing atoms
            while parsing. True by default.

    Returns:
        atom_array (atomworks AtomArray): The atom array for the first
            bioassembly if present, otherwise the atom array for the asymmetric
            unit.
    """
    _, atom_array = load_first_assembly_parsed_and_atom_array(
        structure_path,
        add_missing_atoms = add_missing_atoms
    )

    return atom_array

def save_nucleic_acid_chains_from_structure(
    input_structure_path, 
    output_directory
):
    """
    Given a structure file path, extracts the nucleic acid chains from the
    structure and saves them as separate CIF files in the specified output
    directory.

    Args:
        input_structure_path (str): The path to the input structure file.
        output_directory (str): The directory where the extracted nucleic acid 
            chain files will be saved.

    Side Effects:
        Saves the extracted nucleic acid chains as separate CIF files in the
        specified output directory. The files will be named as
        "output_directory/PDBID[1:3]/PDBID_CHAINID.cif".
    """
    # Setup path objects.
    input_structure_path = pathlib.Path(input_structure_path)
    output_directory = pathlib.Path(output_directory)

    # Extract PDB ID.
    pdb_id = input_structure_path.name.split(".")[0]
    
    # Load the structure.
    try:
        atom_array = atomworks.io.parser.parse(
            input_structure_path
        )["asym_unit"][0]
    except:  
        atom_array = atomworks.io.parser.parse(
            input_structure_path,
            remove_ccds = ["SPW"] # removed for edge case with 1v9g
        )["asym_unit"][0]

    # Extract nucleic acid chain IDs.
    nucleic_acid_mask = np.isin(
        atom_array.chain_type,
        (ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID)
    )
    nucleic_acid_chain_ids = set(atom_array.chain_id[nucleic_acid_mask])

    # Setup the output directory.
    chain_output_directory = output_directory / pdb_id[1:3]
    chain_output_directory.mkdir(parents = True, exist_ok = True)

    # Save each nucleic acid chain as a separate CIF file.
    for chain_id in nucleic_acid_chain_ids:
        # Subset to the current chain.
        chain_mask = (atom_array.chain_id == chain_id)
        chain_atom_array = atom_array[chain_mask]

        # Create output path and parent directories.
        chain_output_path = chain_output_directory / f"{pdb_id}_{chain_id}.cif"

        # Save the chain as a CIF file.
        atomworks.io.utils.io_utils.to_cif_file(
            chain_atom_array,
            chain_output_path,
            include_nan_coords = False,
            include_entity_poly = False
        )

def stable_sequence_hash(sequence,
                         hash_length = 16):
    """
    Given a sequence, computes a stable SHA256-based hash prefix for use in
    cache and file names.

    Args:
        sequence (str): The sequence to hash.
        hash_length (int): The number of hexadecimal characters to keep.

    Returns:
        sequence_hash (str): The stable hash prefix.
    """
    sequence_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    truncated_sequence_hash = sequence_hash[:hash_length]
    return truncated_sequence_hash

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

    # 1 letter codes for DNA residues.
    dna_restypes = [
        "A",
        "C",
        "G",
        "T",
    ]
    dna_restype_to_int = dict(zip(dna_restypes, range(len(dna_restypes))))

    # Unknown residues.
    rna_unknown_restype = "X"
    dna_unknown_restype = "X"
    protein_unknown_restype = "X"
    dssr_unknown_restype = "?"

    # Chain break characters.
    chain_break_character = "/"
    dssr_chain_break_character = "&"

    # DSSR represents modifications of residues with the lower case of their
    # base residue.
    dssr_modified_restypes = [rna_restype.lower() for rna_restype in rna_restypes]

    # NA-MPNN RNA residue type mapping (lowercase NA-MPNN tokens to standard).
    na_mpnn_rna_restype_to_rna_restype = {
        "b": "A",
        "d": "C",
        "h": "G",
        "u": "U",
        "y": "X"
    }

    # NA-MPNN DNA residue type mapping (lowercase NA-MPNN tokens to standard).
    na_mpnn_dna_restype_to_dna_restype = {
        "a": "A",
        "c": "C",
        "g": "G",
        "t": "T",
        "x": "X"
    }

    # Character sets for classifying chain segments from NA-MPNN FASTA output.
    na_mpnn_dna_chars = set("acgtx")
    na_mpnn_rna_chars = set("bdhuy")
    na_mpnn_protein_chars = set("ACDEFGHIKLMNPQRSTVWYX")
    na_mpnn_na_chars = na_mpnn_dna_chars.union(na_mpnn_rna_chars)

    # Protein 3-letter residue name to 1-letter code mapping.
    protein_resname_to_one_letter = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "UNK": "X",
    }

    # DNA 3-letter residue name to 1-letter code mapping.
    dna_resname_to_one_letter = {
        "DA": "A", "DC": "C", "DG": "G", "DT": "T", "DX": "X",
    }

    # DNA 1-letter code to 3-letter residue name mapping.
    dna_one_letter_to_resname = {
        one_letter: resname
        for resname, one_letter in dna_resname_to_one_letter.items()
    }

    # RNA 3-letter residue name to 1-letter code mapping.
    rna_resname_to_one_letter = {
        "A": "A", "C": "C", "G": "G", "U": "U", "RX": "X",
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

    dna_restype_to_rna_restype = dict()
    if na_mpnn_na_shared_tokens:
        na_mpnn_restype_to_int["A"] = na_mpnn_restype_to_int["DA"]
        na_mpnn_restype_to_int["C"] = na_mpnn_restype_to_int["DC"]
        na_mpnn_restype_to_int["G"] = na_mpnn_restype_to_int["DG"]
        na_mpnn_restype_to_int["U"] = na_mpnn_restype_to_int["DT"]
        na_mpnn_restype_to_int["RX"] = na_mpnn_restype_to_int["DX"]

        dna_restype_to_rna_restype["DA"] = "A"
        dna_restype_to_rna_restype["DC"] = "C"
        dna_restype_to_rna_restype["DG"] = "G"
        dna_restype_to_rna_restype["DT"] = "U"
        dna_restype_to_rna_restype["DX"] = "RX"
    
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

    noncanonical_na_resname_to_canonical_resname = {
        "PGP": {
            ChainType.RNA: "G",
        },
        "CH": {
            ChainType.RNA: "C",
        },
        "CBR": {
            ChainType.DNA: "DC",
        },
        "BRU": {
            ChainType.DNA: "DT",
        },
        "FHU": {
            ChainType.RNA: "U",
        },
        "5CM": {
            ChainType.DNA: "DC",
        },
        "A23": {
            ChainType.RNA: "A",
        },
        "6MA": {
            ChainType.DNA: "DA",
        },
        # 3DR is abasic (1',2'-dideoxyribose-5'-phosphate); DA is not a correct
        # cast but is only used at positions that will be redesigned, so the
        # input restype does not matter for NA-MPNN.
        "3DR": {
            ChainType.DNA: "DA",
        },
    }

    dna_backbone_atom_names = {
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
    }
    rna_backbone_atom_names = dna_backbone_atom_names.union({"O2'"})

    na_resname_to_allowed_atom_names = {
        "DA": dna_backbone_atom_names.union(
            {"N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"}
        ),
        "DC": dna_backbone_atom_names.union(
            {"N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"}
        ),
        "DG": dna_backbone_atom_names.union(
            {"N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"}
        ),
        "DT": dna_backbone_atom_names.union(
            {"N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", "C7"}
        ),
        "A": rna_backbone_atom_names.union(
            {"N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"}
        ),
        "C": rna_backbone_atom_names.union(
            {"N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"}
        ),
        "G": rna_backbone_atom_names.union(
            {"N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"}
        ),
        "U": rna_backbone_atom_names.union(
            {"N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"}
        ),
    }

################################################################################
# Sequence and Structure Standardization
################################################################################
def require_chain_type_enum(chain_type):
    """
    Checks that a chain type is already normalized to a ChainType enum.

    Args:
        chain_type (ChainType): The chain type to validate.

    Side Effects:
        Raises a TypeError if the chain type is not a ChainType enum.
    """
    if not isinstance(chain_type, ChainType):
        raise TypeError(
            "Chain type must be a ChainType enum. Normalize inputs "
            "before calling this function."
        )

def require_na_chain_type(chain_type):
    """
    Checks that a chain type is a nucleic acid ChainType enum.

    Args:
        chain_type (ChainType): The chain type to validate.

    Side Effects:
        Raises a TypeError or ValueError if the chain type is invalid.
    """
    require_chain_type_enum(chain_type)
    if chain_type not in (
        ChainType.DNA,
        ChainType.RNA,
        ChainType.DNA_RNA_HYBRID
    ):
        raise ValueError(
            "Nucleic acid sequence data must use DNA, RNA, or DNA/RNA hybrid "
            "chain types."
        )

def check_na_sequence_validity(na_sequence_data,
                               unknown_residue_allowed = False):
    """
    Given nucleic acid sequence data, checks that the sequences are valid.

    Args:
        na_sequence_data ((str, ChainType) list): The nucleic
            sequence data.
        unknown_residue_allowed (bool): Whether unknown residues are allowed.

    Side Effects:
        Raises a ValueError if the nucleic acid sequence data is invalid.
    """
    if len(na_sequence_data) == 0:
        raise ValueError("Nucleic acid sequence data must not be empty.")

    for chain_sequence_data in na_sequence_data:
        if len(chain_sequence_data) != 2:
            raise ValueError(
                "Each nucleic acid sequence entry must contain exactly two "
                "elements: sequence and chain type."
            )

        sequence, chain_type = chain_sequence_data
        require_na_chain_type(chain_type)

        if chain_type == ChainType.RNA:
            valid_chars = set(NAConstants.rna_restypes)
            if unknown_residue_allowed:
                valid_chars.add(NAConstants.rna_unknown_restype)
        elif chain_type == ChainType.DNA:
            valid_chars = set(NAConstants.dna_restypes)
            if unknown_residue_allowed:
                valid_chars.add(NAConstants.dna_unknown_restype)
        else:
            valid_chars = set(NAConstants.rna_restypes)
            valid_chars.update(NAConstants.dna_restypes)
            if unknown_residue_allowed:
                valid_chars.add(NAConstants.rna_unknown_restype)
                valid_chars.add(NAConstants.dna_unknown_restype)

        for c in sequence:
            if c not in valid_chars:
                raise ValueError(
                    f"Invalid character in {chain_type.name} sequence: {c}"
                )

def check_protein_sequence_validity(protein_sequences,
                                    unknown_residue_allowed = False):
    """
    Given protein sequences, checks that the sequences are valid.

    Args:
        protein_sequences (str list): The protein sequences.
        unknown_residue_allowed (bool): Whether unknown residues are allowed.

    Side Effects:
        Raises a ValueError if a protein sequence is invalid.
    """
    valid_chars = set(NAConstants.na_mpnn_protein_chars)
    if not unknown_residue_allowed:
        valid_chars.discard(NAConstants.protein_unknown_restype)

    for protein_sequence in protein_sequences:
        for c in protein_sequence:
            if c not in valid_chars:
                raise ValueError(
                    f"Invalid character in protein sequence: {c}"
                )

def standardize_na_sequence(na_sequence_data,
                            method = None):
    """
    Given nucleic acid sequence data, standardizes the sequence data to a
    canonical representation as a list of (sequence, ChainType) tuples.

    Args:
        na_sequence_data ((str, ChainType) list): The nucleic
            acid sequence data to standardize.
        method (str): The method to use for sequence standardization.
            Options:
                "na_mpnn": Standardize the sequence using the NA-MPNN residue
                    type mapping.
                "dssr": Standardize the sequence using the DSSR unknown residue
                    representation.
                None: no standardization.

    Returns:
        standard_na_sequence_data ((str, ChainType) list): The standardized
            nucleic acid sequence data.
    """
    standard_na_sequence_data = []
    for chain_sequence_data in na_sequence_data:
        if len(chain_sequence_data) != 2:
            raise ValueError(
                "Each nucleic acid sequence entry must contain exactly two "
                "elements: sequence and chain type."
            )

        sequence, chain_type = chain_sequence_data
        require_na_chain_type(chain_type)

        standard_sequence = []
        for c in sequence:
            if method == "na_mpnn":
                if (
                    chain_type in (ChainType.DNA, ChainType.DNA_RNA_HYBRID) and
                    c in NAConstants.na_mpnn_dna_restype_to_dna_restype
                ):
                    standard_sequence.append(
                        NAConstants.na_mpnn_dna_restype_to_dna_restype[c]
                    )
                elif (
                    chain_type in (ChainType.RNA, ChainType.DNA_RNA_HYBRID) and
                    c in NAConstants.na_mpnn_rna_restype_to_rna_restype
                ):
                    standard_sequence.append(
                        NAConstants.na_mpnn_rna_restype_to_rna_restype[c]
                    )
                else:
                    standard_sequence.append(c)
            elif method == "dssr":
                if (
                    c == NAConstants.dssr_unknown_restype or
                    c in NAConstants.dssr_modified_restypes
                ):
                    standard_sequence.append(
                        NAConstants.rna_unknown_restype 
                        if chain_type == ChainType.RNA 
                        else NAConstants.dna_unknown_restype
                    )
                else:
                    standard_sequence.append(c)
            else:
                standard_sequence.append(c)

        standard_na_sequence_data.append(
            ("".join(standard_sequence), chain_type)
        )

    check_na_sequence_validity(
        standard_na_sequence_data,
        unknown_residue_allowed = True
    )

    return standard_na_sequence_data

def extract_sequences_from_structure(structure_path):
    """
    Given a structure file path, extracts nucleic acid and protein sequences
    from the structure using AtomWorks.

    Args:
        structure_path (str): The path to the structure file.

    Returns:
        na_sequence_data ((str, ChainType) list): Standardized nucleic acid
            sequence data extracted from the structure.
        protein_sequences (str list): Protein sequences extracted from the
            structure.
    """
    atom_array = load_first_assembly_atom_array(
        structure_path,
        add_missing_atoms = False
    )

    # Check that chain_id and chain_iid have same unique number.
    if (
        len(np.unique(atom_array.chain_id)) !=
        len(np.unique(atom_array.chain_iid))
    ):
        raise ValueError(
            "Number of unique chain IDs does not match number of unique chain"
            " IIDs. This indicates a symmetry or assembly issue."
        )

    # Iterate over chains in input order.
    ordered_chain_ids = list(dict.fromkeys(atom_array.chain_id.tolist()))

    na_sequence_data = []
    protein_sequences = []
    for chain_id in ordered_chain_ids:
        chain_mask = (atom_array.chain_id == chain_id)
        chain_atom_array = atom_array[chain_mask]

        # Determine chain type.
        chain_types = np.unique(chain_atom_array.chain_type)
        if len(chain_types) != 1:
            raise ValueError(
                f"Multiple chain types found for chain {chain_id} in structure "
                f"{structure_path}: {chain_types}"
            )
        chain_type = ChainType.as_enum(chain_types[0])

        chain_id = chain_atom_array.chain_id[0]

        # Raise an error if the chain type is not one of the supported types.
        if chain_type not in (
            ChainType.POLYPEPTIDE_L,
            ChainType.DNA,
            ChainType.RNA,
            ChainType.DNA_RNA_HYBRID
        ):
            raise ValueError(
                f"Unsupported chain type {chain_type} for chain"
                f" {chain_id} in structure {structure_path}"
            )

        # Extract residue names using token starts.
        token_starts = get_token_starts(chain_atom_array)
        token_ends = list(token_starts[1:]) + [len(chain_atom_array)]

        sequence = []
        for token_start, token_end in zip(token_starts, token_ends):
            token_atom_array = chain_atom_array[token_start:token_end]
            token_res_names = np.unique(token_atom_array.res_name)

            if len(token_res_names) != 1:
                raise ValueError(
                    f"Multiple residue names found for token starting at index "
                    f"{token_start} in chain {chain_id} in structure "
                    f"{structure_path}: {token_res_names}"
                )
            token_res_name = token_res_names[0]

            # Convert the residue name to a 1-letter code. Non-canonical
            # residues are mapped to the unknown residue code.
            if chain_type == ChainType.POLYPEPTIDE_L:
                one_letter = NAConstants.protein_resname_to_one_letter.get(
                    token_res_name,
                    NAConstants.protein_unknown_restype
                )
            elif chain_type == ChainType.DNA:
                one_letter = NAConstants.dna_resname_to_one_letter.get(
                    token_res_name,
                    NAConstants.dna_unknown_restype
                )
            elif chain_type == ChainType.RNA:
                one_letter = NAConstants.rna_resname_to_one_letter.get(
                    token_res_name,
                    NAConstants.rna_unknown_restype
                )
            else:
                if token_res_name in NAConstants.dna_resname_to_one_letter:
                    one_letter = NAConstants.dna_resname_to_one_letter[
                        token_res_name
                    ]
                elif token_res_name in NAConstants.rna_resname_to_one_letter:
                    one_letter = NAConstants.rna_resname_to_one_letter[
                        token_res_name
                    ]
                else:
                    one_letter = NAConstants.dna_unknown_restype

            sequence.append(one_letter)

        sequence = "".join(sequence)
        if chain_type == ChainType.POLYPEPTIDE_L:
            protein_sequences.append(sequence)
        else:
            na_sequence_data.append((sequence, chain_type))

    na_sequence_data = standardize_na_sequence(na_sequence_data)
    if len(na_sequence_data) == 0:
        raise ValueError("No nucleic acid chains found in structure.")

    return na_sequence_data, protein_sequences

def remove_protein_chains_from_structure(structure_path):
    """
    Given a structure file path, creates a temporary PDB file containing only
    the nucleic acid chains (DNA, RNA, DNA_RNA_HYBRID) from the structure.

    Args:
        structure_path (str): The path to the input structure file.

    Returns:
        temp_pdb_path (str): The path to the temporary PDB file containing
            only nucleic acid chains.
        temp_directory (tempfile.TemporaryDirectory): The temporary directory
            that owns the returned file path. The caller is responsible for
            cleaning it up when it is no longer needed.
    """
    atom_array = load_first_assembly_atom_array(
        structure_path,
        add_missing_atoms = False
    )

    # Filter to nucleic acid chains only.
    na_mask = np.isin(
        atom_array.chain_type,
        (ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID)
    )
    na_atom_array = atom_array[na_mask]

    if len(na_atom_array) == 0:
        raise ValueError(
            f"No nucleic acid chains found in structure: {structure_path}"
        )

    # Preserve the original structure stem in the temporary file name so
    # downstream tools keep emitting stable structure-based output names.
    structure_stem = pathlib.Path(structure_path).stem
    temp_directory = tempfile.TemporaryDirectory()
    temp_pdb_path = os.path.join(temp_directory.name, f"{structure_stem}.pdb")
    pdb_string = atomworks.io.utils.io_utils.to_pdb_string(na_atom_array)
    write_text_file(temp_pdb_path, pdb_string)
    
    return temp_pdb_path, temp_directory

def canonicalize_noncanonical_na_residues(structure_path):
    """
    Given a structure file path, rewrites non-canonical nucleic acid residues
    to their canonical equivalents by renaming the residue and stripping atoms
    that do not belong to the canonical form. If the structure contains no
    non-canonical residues, the original path is returned unchanged.

    Args:
        structure_path (str): The path to the input structure file.

    Returns:
        output_structure_path (str): The path to the structure file with
            canonicalized residues. This is either the original path (if no
            changes were needed) or a path inside a new temporary directory.
        temp_directory (tempfile.TemporaryDirectory or None): The temporary
            directory that owns the returned file path, or None if no changes
            were made. The caller is responsible for cleaning it up when it is
            no longer needed.
    """
    # Load the structure.
    atom_array = load_first_assembly_atom_array(
        structure_path,
        add_missing_atoms = False
    )

    # Iterate over tokens and rewrite non-canonical residues.
    token_starts = get_token_starts(atom_array)
    token_ends = list(token_starts[1:]) + [len(atom_array)]
    keep_mask = np.ones(len(atom_array), dtype = bool)
    output_atom_array = atom_array.copy()
    structure_changed = False
    for token_start, token_end in zip(token_starts, token_ends):
        # Grab the atom array for the current token.
        token_atom_array = atom_array[token_start : token_end]

        # Determine the chain type for the token, skip non-NA tokens.
        token_chain_types = np.unique(token_atom_array.chain_type)
        if len(token_chain_types) != 1:
            raise ValueError(
                "Each token must have exactly one chain type when rewriting "
                "non-canonical nucleic acid residues."
            )
        token_chain_type = ChainType.as_enum(token_chain_types[0])
        if token_chain_type == ChainType.DNA_RNA_HYBRID:
            if np.any(token_atom_array.atom_name == "O2'"):
                token_chain_type = ChainType.RNA
            else:
                token_chain_type = ChainType.DNA
        if token_chain_type not in (ChainType.DNA, ChainType.RNA):
            continue
        
        # Determine the residue name for the token.
        token_res_names = np.unique(token_atom_array.res_name)
        if len(token_res_names) != 1:
            raise ValueError(
                "Each token must have exactly one residue name when rewriting "
                "non-canonical nucleic acid residues."
            )
        token_res_name = token_res_names[0]

        # Skip canonical residues.
        if token_chain_type == ChainType.DNA and \
           token_res_name in NAConstants.dna_resname_to_one_letter:
            continue
        if token_chain_type == ChainType.RNA and \
           token_res_name in NAConstants.rna_resname_to_one_letter:
            continue
        
        # Map the non-canonical residue name to a canonical residue name, and
        # skip if there is no known mapping for this residue and chain type.
        target_res_name = \
            NAConstants.noncanonical_na_resname_to_canonical_resname.get(
                token_res_name,
                {}
            ).get(token_chain_type)
        if target_res_name is None:
            continue
        
        # Determine which atoms to keep based on the target residue name.
        allowed_atom_names = NAConstants.na_resname_to_allowed_atom_names[
            target_res_name
        ]
        token_keep_mask = np.isin(
            token_atom_array.atom_name,
            list(allowed_atom_names)
        )
        if not np.any(token_keep_mask):
            raise ValueError(
                "Canonicalization removed every atom from a non-canonical NA "
                f"residue: {token_res_name}."
            )

        # Mark atoms for removal and update residue names and hetero flags for
        # kept atoms.
        token_global_indices = np.arange(token_start, token_end)
        kept_token_indices = token_global_indices[token_keep_mask]
        keep_mask[token_global_indices[np.logical_not(token_keep_mask)]] = False
        output_atom_array.res_name[kept_token_indices] = target_res_name
        output_atom_array.hetero[kept_token_indices] = False
        structure_changed = True

    # If no changes were made, return the original path.
    if not structure_changed:
        return structure_path, None

    # Clear hetero flags on all NA atoms and apply the keep mask.
    na_mask = np.isin(
        output_atom_array.chain_type,
        (ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID)
    )
    output_atom_array.hetero[na_mask] = False
    output_atom_array = output_atom_array[keep_mask]

    # Preserve the original structure stem in the temporary file name so
    # downstream tools keep emitting stable structure-based output names.
    structure_stem = pathlib.Path(structure_path).stem
    temp_directory = tempfile.TemporaryDirectory()
    output_structure_path = os.path.join(
        temp_directory.name,
        f"{structure_stem}.pdb"
    )
    pdb_string = atomworks.io.utils.io_utils.to_pdb_string(output_atom_array)
    write_text_file(output_structure_path, pdb_string)

    return output_structure_path, temp_directory

def prepare_complex_sequence_data(na_sequence_data = None,
                                  protein_sequences = None):
    """
    Validates pre-standardized nucleic acid/protein sequence data and derives
    common sequence-context flags for design and scoring code paths.

    Args:
        na_sequence_data ((str, ChainType) list): Standardized nucleic acid
            sequence data.
        protein_sequences (str list): The protein sequences.

    Returns:
        result (dict): A dictionary containing derived flags.
    """
    if na_sequence_data is None or protein_sequences is None:
        raise ValueError(
            "na_sequence_data and protein_sequences are required."
        )

    check_na_sequence_validity(
        na_sequence_data,
        unknown_residue_allowed = True
    )
    if len(na_sequence_data) == 0:
        raise ValueError("No nucleic acid chains found in sequence data.")
    
    protein_sequences = list(protein_sequences)
    check_protein_sequence_validity(
        protein_sequences,
        unknown_residue_allowed = True
    )

    na_chain_types = [
        chain_type
        for _, chain_type in na_sequence_data
    ]
    num_na_residues = sum(
        len(sequence) for sequence, _ in na_sequence_data
    )
    has_protein = len(protein_sequences) > 0
    has_dna = any(
        chain_type in (ChainType.DNA, ChainType.DNA_RNA_HYBRID)
        for chain_type in na_chain_types
    )
    has_rna = any(
        chain_type in (ChainType.RNA, ChainType.DNA_RNA_HYBRID)
        for chain_type in na_chain_types
    )
    is_single_rna_chain = (
        len(na_sequence_data) == 1 and
        na_chain_types[0] == ChainType.RNA
    )
    is_single_dna_chain = (
        len(na_sequence_data) == 1 and
        na_chain_types[0] == ChainType.DNA
    )
    is_monomer_rna = is_single_rna_chain and not has_protein

    result = {
        "na_chain_types": na_chain_types,
        "num_na_residues": num_na_residues,
        "has_protein": has_protein,
        "has_dna": has_dna,
        "has_rna": has_rna,
        "is_single_rna_chain": is_single_rna_chain,
        "is_single_dna_chain": is_single_dna_chain,
        "is_monomer_rna": is_monomer_rna,
    }
    return result

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
                "ribonanzanet": Convert ARNIE/RibonanzaNet lettered
                    pseudoknot symbols to the repo's default convention.
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
        elif method == "ribonanzanet" and c.isalpha():
            standard_secondary_structure.append(c.swapcase())
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
    check_na_sequence_validity(
        [(sequence, ChainType.RNA)],
        unknown_residue_allowed = False
    )

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
    check_na_sequence_validity(
        [(sequence, ChainType.RNA)],
        unknown_residue_allowed = False
    )
    
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
    check_na_sequence_validity(
        [(sequence, ChainType.RNA)],
        unknown_residue_allowed = False
    )
    
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
def calculate_min_cross_chain_pae(chain_pair_pae_min_matrix,
                                  chain_sequence_data):
    """
    Given the chain_pair_pae_min matrix from AlphaFold3 summary confidences,
    computes the minimum PAE across protein-to-nucleic-acid chain pairs.

    Args:
        chain_pair_pae_min_matrix (list of lists): The NxN chain_pair_pae_min
            matrix from AlphaFold3 summary_confidences.json, where N is the
            number of chains.
        chain_sequence_data ((str, ChainType) list): The chain sequence data
            used to build the AlphaFold3 input.

    Returns:
        min_cross_chain_pae (float or None): The minimum protein-to-NA value in
            the chain_pair_pae_min matrix, or None if there is no protein/NA
            interface in the input.
    """
    matrix = np.array(chain_pair_pae_min_matrix)
    n = matrix.shape[0]

    if matrix.shape != (n, n):
        raise ValueError("chain_pair_pae_min matrix must be square.")
    if n != len(chain_sequence_data):
        raise ValueError(
            "chain_pair_pae_min matrix size must match the number of chains."
        )

    protein_mask = np.array(
        [
            chain_type == ChainType.POLYPEPTIDE_L
            for _, chain_type in chain_sequence_data
        ],
        dtype = bool
    )
    na_mask = np.array(
        [
            chain_type in (
                ChainType.DNA,
                ChainType.RNA,
                ChainType.DNA_RNA_HYBRID
            )
            for _, chain_type in chain_sequence_data
        ],
        dtype = bool
    )

    if not np.any(protein_mask) or not np.any(na_mask):
        return None

    protein_to_na_mask = np.logical_or(
        np.outer(protein_mask, na_mask),
        np.outer(na_mask, protein_mask)
    )
    protein_to_na_values = matrix[protein_to_na_mask]

    if protein_to_na_values.size == 0:
        return None

    return float(np.min(protein_to_na_values))

def run_alphafold3(name, 
                   chain_sequence_data,
                   output_dir,
                   num_diffusion_samples = 5,
                   num_seeds = 1,
                   fixed_seeds = None,
                   run_data_pipeline = False,
                   run_inference = True,
                   precomputed_chain_data = None,
                   buckets = "128,256,512",
                   flash_attention_implementation = "triton",
                   alphafold3_apptainer_path = "/software/containers/users/ikalvet/mlfold3/mlfold3_01.sif",
                   alphafold3_path = "/opt/alphafold3/run_alphafold.py",
                   model_dir = "/databases/alphafold",
                   db_dir = "/databases/lab/af3_DB",):
    """
    Given a name, chain sequence data, and an output directory, runs
    AlphaFold3 to predict the structure of the complex.

    Args:
        name (str): A name of the complex.
        chain_sequence_data ((str, ChainType) list): A list of tuples, where
            each tuple contains the sequence and chain type for a chain.
        output_dir (str): The path to the output directory.
        num_diffusion_samples (int): The number of diffusion samples to 
            generate. Default is 5.
        num_seeds (int): The number of model seeds to generate and use. Default
            is 1. This argument is mutually exclusive with fixed_seeds.
        fixed_seeds (int list): A list of fixed seeds to use for the model. This
            argument is mutually exclusive with num_seeds.
        run_data_pipeline (bool): Whether to run the data pipeline (whether to
            perform the MSA and templates searches). This argument is mutually
            exclusive with run_inference.
        run_inference (bool): Whether to run model inference. Set to False to
            run only the data pipeline (MSA and template search) without
            folding. Default is True. This argument is mutually exclusive with
            run_data_pipeline.
        precomputed_chain_data (dict): A dictionary mapping each protein
            sequence (str) to a dictionary containing pre-computed MSA and
            template data with keys "unpairedMsa", "pairedMsa", and
            "templates". Repeated protein sequences reuse the same data for
            each occurrence. When provided, these fields are injected into
            matching protein sequence entries of the input JSON. Default is
            None.
        buckets (str): A comma separated list of integers. Strictly increasing 
            order of token sizes for which to cache compilations. For any input 
            with more tokens than the largest bucket size, a new bucket is 
            created for exactly that number of tokens. The alphafold3 default 
            is "256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120".
        flash_attention_implementation (str): The flash attention 
            implementation to use.
        alphafold3_apptainer_path (str): The path to the AlphaFold3 apptainer
            for running AlphaFold3.
        alphafold3_path (str): The path to the AlphaFold3 run file.
        model_dir (str): The path to the AlphaFold3 model directory.
        db_dir (str): The path to the AlphaFold3 database bundle used for the
            data pipeline.
    
    Returns:
        result (dict): A dictionary containing:
            json_input_path (str): The path to the input/output JSON file
                (contains MSA/template data after data pipeline runs).
        Only present if run_inference is True:
            predicted_structure_path (str): The path to the predicted structure
                file.
            predicted_confidences_path (str): The path to the predicted
                confidences file.
            summary_confidences_path (str): The path to the summary confidences
                file.
            ptm (float): The predicted PTM score.
            iptm (float or None): The predicted iPTM score. None for single
                chain predictions.
            plddt (float): The predicted pLDDT score.
            pae (float): The predicted pAE score.
            chain_pair_pae_min (list of lists): The NxN chain pair minimum PAE
                matrix.
            min_cross_chain_pae (float or None): The minimum protein-to-NA
                chain pair value from chain_pair_pae_min.
    """
    # Check that both num_seeds and fixed_seeds are not set.
    if num_seeds is not None and fixed_seeds is not None:
        raise ValueError("Both num_seeds and fixed_seeds cannot be set at the same time.")
    if run_data_pipeline and run_inference:
        raise ValueError(
            "run_data_pipeline and run_inference cannot both be True."
        )

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
        model_seeds = [
            int(seed_rng.integers(0, 2 ** 32 - 1))
            for _ in range(num_seeds)
        ]

    # Prepare the sequences input.
    sequences_input = []
    for i, chain_sequence_data_entry in enumerate(chain_sequence_data):
        if len(chain_sequence_data_entry) != 2:
            raise ValueError(
                "Each chain sequence entry must contain exactly two elements: "
                "sequence and chain type."
            )

        sequence, chain_type = chain_sequence_data_entry
        require_chain_type_enum(chain_type)

        if chain_type == ChainType.POLYPEPTIDE_L:
            polytype = "protein"
        elif chain_type == ChainType.RNA:
            check_na_sequence_validity([(sequence, chain_type)])
            polytype = "rna"
        elif chain_type == ChainType.DNA:
            check_na_sequence_validity([(sequence, chain_type)])
            polytype = "dna"
        elif chain_type == ChainType.DNA_RNA_HYBRID:
            raise ValueError(
                "Unable to run AlphaFold3 on DNA/RNA hybrid chains"
            )
        else:
            raise ValueError(
                f"Unsupported chain type for AlphaFold3 input: {chain_type}"
            )

        sequences_entry_dict = {
            polytype: {
                "id": chain_num_to_chain_id(i),
                "sequence": sequence,
            }
        }

        # If running inference, set up the MSA and template fields.
        if run_inference:
            if chain_type == ChainType.POLYPEPTIDE_L:
                sequences_entry_dict[polytype]["unpairedMsa"] = ""
                sequences_entry_dict[polytype]["pairedMsa"] = ""
                sequences_entry_dict[polytype]["templates"] = []

                # Add pre-computed MSA and template data if available.
                if precomputed_chain_data is not None:
                    chain_data = precomputed_chain_data[sequence]

                    sequences_entry_dict[polytype]["unpairedMsa"] = \
                        chain_data["unpairedMsa"]
                    sequences_entry_dict[polytype]["pairedMsa"] = \
                        chain_data["pairedMsa"]
                    sequences_entry_dict[polytype]["templates"] = \
                        chain_data["templates"]
            elif chain_type == ChainType.RNA:
                sequences_entry_dict[polytype]["unpairedMsa"] = ""

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
                f"--db_dir={db_dir}",
                f"--run_data_pipeline={run_data_pipeline}",
                f"--run_inference={run_inference}",
                f"--buckets={buckets}",
                f"--num_diffusion_samples={num_diffusion_samples}",
                f"--output_dir={output_dir}",
                f"--json_path={temp_json_file.name}",
                f"--flash_attention_implementation={flash_attention_implementation}",
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

    # Check that the data JSON file exists (always produced).
    if not os.path.exists(json_input_path):
        raise ValueError(f"Output JSON file not found: {json_input_path}")

    # If inference was not run, return only the data JSON path.
    if not run_inference:
        result = {
            "json_input_path": json_input_path,
        }
        return result

    predicted_structure_path = os.path.join(name_output_directory, f"{name}_model.cif")
    predicted_confidences_path = os.path.join(name_output_directory, f"{name}_confidences.json")
    summary_confidences_path = os.path.join(name_output_directory, f"{name}_summary_confidences.json")

    # Check that the output files exist.
    if not os.path.exists(predicted_structure_path):
        raise ValueError(f"Predicted structure file not found: {predicted_structure_path}")
    if not os.path.exists(predicted_confidences_path):
        raise ValueError(f"Predicted confidences file not found: {predicted_confidences_path}")
    if not os.path.exists(summary_confidences_path):
        raise ValueError(f"Summary confidences file not found: {summary_confidences_path}")
    
    # Extract confidence scores.
    summary_confidences_dict = read_json_file(summary_confidences_path)
    ptm = summary_confidences_dict["ptm"]
    iptm = summary_confidences_dict["iptm"]
    chain_pair_pae_min = summary_confidences_dict["chain_pair_pae_min"]
    min_cross_chain_pae = calculate_min_cross_chain_pae(
        chain_pair_pae_min,
        chain_sequence_data
    )

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
        "pae": pae,
        "iptm": iptm,
        "chain_pair_pae_min": chain_pair_pae_min,
        "min_cross_chain_pae": min_cross_chain_pae
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
        tmp_directory = None
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
        if tmp_directory is not None:
            tmp_directory.cleanup()
        
        return result
    except (subprocess.CalledProcessError, ValueError) as e:
        # Clean up the temporary directory if it was created.
        if tmp_directory is not None:
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
        tmp_directory = None
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
        if tmp_directory is not None:
            tmp_directory.cleanup()
        
        return result
    except (subprocess.CalledProcessError, ValueError) as e:
        # Clean up the temporary directories.
        tmp_intermediate_directory.cleanup()
        if tmp_directory is not None:
            tmp_directory.cleanup()
        raise e

def run_rclamps(structure_path,
                output_directory = None,
                domain_type = None,
                rclamps_apptainer_path = "/software/containers/users/akubaney/rclamps.sif",
                rclamps_directory = "/home/akubaney/software/rCLAMPS"):
    """
    Given a path to a structure, runs the rCLAMPS algorithm on the protein
    sequence of the structure to predict the DNA specificity of the structure. 
    The predicted PPM is aligned against the ground truth DNA chains of the
    structure, so that the output shape matches run_na_mpnn_specificity /
    run_deeppbs.

    Args:
        structure_path (str): The path to the tertiary structure file.
        output_directory (str): The path to the output directory. If None, a
            temporary directory will be created.
        domain_type (str): The rCLAMPS domain type. Options are "zf-C2H2" and
            "homeodomain".
        rclamps_apptainer_path (str): The path to the rCLAMPS apptainer.
        rclamps_directory (str): The path to the rCLAMPS root directory.

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
    structure_path = os.path.abspath(structure_path)
    if not os.path.exists(structure_path):
        raise ValueError(f"Invalid structure path: {structure_path}")

    if output_directory is None:
        tmp_directory = tempfile.TemporaryDirectory()
        output_directory = tmp_directory.name
    else:
        tmp_directory = None
        output_directory = os.path.abspath(output_directory)

    if domain_type not in ("zf-C2H2", "homeodomain"):
        raise ValueError(f"Invalid rCLAMPS domain type: {domain_type}")

    structure_name = os.path.splitext(os.path.basename(structure_path))[0]

    specificity_output_directory = os.path.join(output_directory, "specificity")
    if not os.path.exists(specificity_output_directory):
        os.makedirs(specificity_output_directory)

    # Extract sequences from the structure. Use the first protein chain as
    # input to rCLAMPS.
    na_sequence_data, protein_sequences = extract_sequences_from_structure(structure_path)
    if len(protein_sequences) == 0:
        raise ValueError(f"No protein chains found in structure: {structure_path}")
    protein_sequence = protein_sequences[0]

    # rCLAMPS's HMMER 2.3.2 parser truncates IDs to 10 characters in the
    # alignment section, so pad short structure names to avoid a parser bug.
    protein_id = structure_name + "_rclamps"

    tmp_intermediate_directory = tempfile.TemporaryDirectory()
    try:
        # For C2H2 zinc fingers, rCLAMPS requires the hmmer table output of
        # findDomains.py.
        if domain_type == "zf-C2H2":
            # Create the hmmer input fasta file.
            protein_fasta_path = os.path.join(tmp_intermediate_directory.name, "protein.fa")
            write_text_file(protein_fasta_path, f">{protein_id}\n{protein_sequence}\n")

            # Create the expected output path for the hmmer table.
            hmmer_table_path = os.path.join(tmp_intermediate_directory.name, "hmmer_out.txt")

            # Run findDomains.py to get the hmmer table output.
            subprocess.run(
                [
                    "apptainer", 
                    "exec",
                    "--pwd", 
                    os.path.join(rclamps_directory, "cis_bp", "C2H2-ZF"),
                    rclamps_apptainer_path,
                    "python", 
                    "findDomains.py",
                    "--input_fasta", 
                    protein_fasta_path,
                    "--output_file", 
                    hmmer_table_path
                ],
                check = True,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL
            )

            # Create the input file for predictionExample.py.
            prot_seq_input_path = os.path.join(
                tmp_intermediate_directory.name, 
                "prot_seq.txt"
            )

            # Read the hmmer table, extract the prot and coreSeq columns.
            hmmer_table = pd.read_csv(hmmer_table_path, sep="\t")
            prot_seq_lines = [
                f"{row['prot']}\t{row['coreSeq']}"
                for _, row in hmmer_table.iterrows()
            ]
            write_text_file(
                prot_seq_input_path,
                "\n".join(prot_seq_lines) + "\n"
            )
        # For homeodomains, rCLAMPS takes the full protein sequence as input.
        elif domain_type == "homeodomain":
            prot_seq_input_path = os.path.join(
                tmp_intermediate_directory.name, 
                "protein.fa"
            )

            write_text_file(
                prot_seq_input_path, 
                f">{protein_id}\n{protein_sequence}\n"
            )

        # Run predictionExample.py to produce predicted_pwms.txt.
        prediction_output_dir = os.path.join(
            tmp_intermediate_directory.name, 
            "prediction_out"
        )
        os.makedirs(prediction_output_dir)

        subprocess.run(
            [
                "apptainer", 
                "exec",
                "--pwd", 
                os.path.join(rclamps_directory, "code"),
                rclamps_apptainer_path,
                "python", 
                "predictionExample.py",
                "--domain_type", 
                domain_type,
                "--protein_file", 
                prot_seq_input_path,
                "--output_dir", 
                prediction_output_dir
            ],
            check = True,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        # Extract the predicted PPM from the output.
        predicted_pwms_path = os.path.join(
            prediction_output_dir,
            "predicted_pwms.txt"
        )
        if not os.path.exists(predicted_pwms_path):
            raise ValueError(
                f"Predicted PPM file not found: {predicted_pwms_path}"
            )

        # Copy the predicted PPM file.
        shutil.copy(
            predicted_pwms_path,
            os.path.join(specificity_output_directory, f"{structure_name}.txt")
        )

        # Parse predicted_pwms.txt into an L x 4 array of probabilities.
        base_to_idx = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3
        }
        ppm_rows = {}
        predicted_ppm_text = read_text_file(predicted_pwms_path).strip()
        predicted_ppm_lines = predicted_ppm_text.split("\n")
        for line in predicted_ppm_lines[1:]:
            _, base_position, base, prob = line.strip().split("\t")
            base_position = int(base_position)
            if base_position not in ppm_rows:
                ppm_rows[base_position] = [0.0] * 4
            ppm_rows[base_position][base_to_idx[base]] = float(prob)
        predicted_ppm = np.array(
            [ppm_rows[i] for i in range(len(ppm_rows))],
            dtype=np.float64
        )

        # Extract the DNA chain sequences from the structure.
        dna_chain_sequences = [
            seq for (seq, chain_type) in na_sequence_data
            if chain_type == ChainType.DNA
        ]
        if len(dna_chain_sequences) == 0:
            raise ValueError(
                f"No DNA chains found in structure: {structure_path}"
            )

        # Construct the S array and chain labels array for alignment.
        S_list = []
        chain_labels_list = []
        for chain_idx, dna_sequence in enumerate(dna_chain_sequences):
            for one_letter_restype in dna_sequence:
                restype = NAConstants.dna_one_letter_to_resname[
                    one_letter_restype
                ]
                S_list.append(NAConstants.na_mpnn_restype_to_int[restype])
                chain_labels_list.append(chain_idx)
        S = np.array(S_list, dtype = np.int32)
        chain_labels = np.array(chain_labels_list, dtype = np.int32)

        # Create the masks for alignment.
        total_len = len(S)
        protein_mask = np.zeros(total_len, dtype = np.int32)
        dna_mask = np.ones(total_len, dtype = np.int32)
        rna_mask = np.zeros(total_len, dtype = np.int32)

        # Compute the reverse complement of the predicted PPM.
        predicted_ppm_rc = np.flip(np.flip(predicted_ppm, axis = 1), axis = 0)

        # Align the predicted PPM and its reverse complement to the true
        # sequence. ppm_mask is 1 at positions the predicted PPM covers, 0
        # elsewhere; use it as the subject mask so scoring skips uncovered
        # positions (rCLAMPS emits a short ~4-12 bp PPM, so most of the DNA
        # sequence has no prediction and an all-zero aligned row).
        aligned_ppm, ppm_mask, _, _, _, _ = align_ppms(
            [(predicted_ppm, "dna"), (predicted_ppm_rc, "dna")],
            S, chain_labels, protein_mask, dna_mask, rna_mask
        )
        mask = ppm_mask

        result = {
            "input_structure_name": structure_name,
            "input_structure_path": structure_path,
            "name": structure_name,
            "predicted_ppm_na_mpnn_format": aligned_ppm,
            "true_sequence_na_mpnn_format": S,
            "chain_labels": chain_labels,
            "mask": mask,
            "protein_mask": protein_mask,
            "dna_mask": dna_mask,
            "rna_mask": rna_mask,
            "encoded_residues": None,
            "encoded_residues_dict": None,
            "specificity_method": "rclamps",
            "model_weights_path": None,
            "num_samples": 1,
            "temperature": None
        }

        tmp_intermediate_directory.cleanup()

        if tmp_directory is not None:
            tmp_directory.cleanup()
        return result

    except (subprocess.CalledProcessError, ValueError) as e:
        tmp_intermediate_directory.cleanup()

        if tmp_directory is not None:
            tmp_directory.cleanup()
        raise e

################################################################################
# Sequence Comparison
################################################################################
def calculate_sequence_recovery(reference_na_sequence_data,
                                subject_na_sequence_data,
                                unknown_residue_allowed_in_reference = False):
    """
    Given reference and subject nucleic acid sequence data, calculates the
    sequence recovery of the subject sequence.

    Args:
        reference_na_sequence_data ((str, ChainType) list): The
            reference nucleic acid sequence data.
        subject_na_sequence_data ((str, ChainType) list): The
            subject nucleic acid sequence data.
        unknown_residue_allowed_in_reference (bool): Whether unknown residues
            are allowed in the reference sequence.

    Returns:
        result (dict): A dictionary containing:
            sequence_recovery (float): The sequence recovery of the sequence.
    """
    check_na_sequence_validity(
        subject_na_sequence_data,
        unknown_residue_allowed = False
    )
    check_na_sequence_validity(
        reference_na_sequence_data,
        unknown_residue_allowed = unknown_residue_allowed_in_reference
    )

    if len(subject_na_sequence_data) != len(reference_na_sequence_data):
        raise ValueError(
            "Number of subject nucleic acid chains must match the number of "
            "reference nucleic acid chains."
        )
        
    for chain_idx, (
        reference_chain_sequence_data,
        subject_chain_sequence_data
    ) in enumerate(
        zip(reference_na_sequence_data, subject_na_sequence_data)
    ):
        reference_chain_sequence, reference_chain_type = \
            reference_chain_sequence_data
        subject_chain_sequence, subject_chain_type = subject_chain_sequence_data

        if subject_chain_type != reference_chain_type:
            raise ValueError(
                f"Subject nucleic acid chain type ({subject_chain_type.name}) "
                f"must match the reference nucleic acid chain type "
                f"({reference_chain_type.name}) "
                f"for chain {chain_idx}."
            )

        if len(subject_chain_sequence) != len(reference_chain_sequence):
            raise ValueError(
                f"Length of subject chain {chain_idx} sequence "
                f"({len(subject_chain_sequence)}) must match length of "
                f"reference chain {chain_idx} sequence "
                f"({len(reference_chain_sequence)})."
            )

    # Calculate the number of correct residues.
    num_correct = 0
    num_residues = 0
    for reference_chain_sequence_data, subject_chain_sequence_data in zip(
        reference_na_sequence_data,
        subject_na_sequence_data
    ):
        reference_chain_sequence, reference_chain_type = \
            reference_chain_sequence_data
        subject_chain_sequence, _ = subject_chain_sequence_data

        for subject_residue, reference_residue in zip(
            subject_chain_sequence,
            reference_chain_sequence
        ):
            # Skip unknown residues in the reference sequence.
            if reference_chain_type == ChainType.RNA:
                reference_is_unknown = (
                    reference_residue == NAConstants.rna_unknown_restype
                )
            elif reference_chain_type == ChainType.DNA:
                reference_is_unknown = (
                    reference_residue == NAConstants.dna_unknown_restype
                )
            elif reference_chain_type == ChainType.DNA_RNA_HYBRID:
                reference_is_unknown = (
                    reference_residue == NAConstants.rna_unknown_restype or
                    reference_residue == NAConstants.dna_unknown_restype
                )
            else:
                raise ValueError(
                    f"Unsupported chain type: {reference_chain_type}"
                )

            if unknown_residue_allowed_in_reference and reference_is_unknown:
                continue
            elif (
                not unknown_residue_allowed_in_reference and 
                reference_is_unknown
            ):
                raise ValueError(
                    "Unknown residues are not allowed in the reference "
                    "sequence, but an unknown residue was found."
                )

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

def calculate_gc_content(na_sequence_data):
    """
    Given nucleic acid sequence data, calculates the GC content across all
    chains combined.

    Args:
        na_sequence_data ((str, ChainType) list): The nucleic acid
            sequence data.

    Returns:
        gc_content (float): The fraction of G and C residues, excluding
            unknown residues (X).
    """
    na_sequence_data = standardize_na_sequence(na_sequence_data)

    gc_count = 0
    total_count = 0

    for chain_sequence, chain_type in na_sequence_data:
        if chain_type == ChainType.RNA:
            unknown_residues = {NAConstants.rna_unknown_restype}
        elif chain_type == ChainType.DNA:
            unknown_residues = {NAConstants.dna_unknown_restype}
        elif chain_type == ChainType.DNA_RNA_HYBRID:
            unknown_residues = {
                NAConstants.rna_unknown_restype,
                NAConstants.dna_unknown_restype
            }
        else:
            raise ValueError(f"Unsupported chain type: {chain_type}")

        for c in chain_sequence:
            if c in unknown_residues:
                continue
            total_count += 1
            if c in ("G", "C"):
                gc_count += 1

    if total_count == 0:
        raise ValueError("No valid residues found for GC content calculation.")

    gc_content = gc_count / total_count
    return gc_content

def calculate_na_c1_rmsd(reference_atom_array,
                         subject_atom_array):
    """
    Given reference and subject atom arrays, extracts nucleic acid C1' atoms,
    superimposes the subject onto the reference using those atoms, and
    computes the RMSD.

    Args:
        reference_atom_array (AtomArray): The reference atom array.
        subject_atom_array (AtomArray): The subject atom array.

    Returns:
        na_c1_rmsd (float): The C1' RMSD after C1'-based superimposition.
    """
    # Extract NA C1' atoms from reference.
    reference_na_mask = np.isin(
        reference_atom_array.chain_type,
        (ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID)
    )
    reference_c1_atom_array = reference_atom_array[
        reference_na_mask & (reference_atom_array.atom_name == "C1'")
    ]

    # Extract NA C1' atoms from subject.
    subject_na_mask = np.isin(
        subject_atom_array.chain_type,
        (ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID)
    )
    subject_c1_atom_array = subject_atom_array[
        subject_na_mask & (subject_atom_array.atom_name == "C1'")
    ]

    if len(reference_c1_atom_array) == 0 or len(subject_c1_atom_array) == 0:
        raise ValueError("No nucleic acid C1' atoms found for RMSD.")
    if len(reference_c1_atom_array) != len(subject_c1_atom_array):
        raise ValueError(
            f"NA C1' atom count mismatch: reference={len(reference_c1_atom_array)}, "
            f"subject={len(subject_c1_atom_array)}."
        )

    # Superimpose using NA C1' atoms, and compute RMSD on the same atoms.
    superimposed, _ = biotite.structure.superimpose(
        reference_c1_atom_array,
        subject_c1_atom_array
    )
    na_c1_rmsd = float(
        biotite.structure.rmsd(
            reference_c1_atom_array,
            superimposed
        )
    )

    return na_c1_rmsd

def calculate_na_c1_lddt_gddt(reference_atom_array,
                              subject_atom_array):
    """
    Given reference and subject atom arrays, extracts nucleic acid C1' atoms
    and computes LDDT and gDDT-like scores on those atoms.

    Args:
        reference_atom_array (AtomArray): The reference atom array.
        subject_atom_array (AtomArray): The subject atom array.

    Returns:
        result (dict): A dictionary containing:
            c1_prime_lddt (float): The C1' LDDT score.
            c1_prime_gddt (float): The C1' gDDT-like score.
    """
    # Extract NA C1' atoms from reference.
    reference_na_mask = np.isin(
        reference_atom_array.chain_type,
        (ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID)
    )
    reference_c1_atom_array = reference_atom_array[
        reference_na_mask & (reference_atom_array.atom_name == "C1'")
    ]

    # Extract NA C1' atoms from subject.
    subject_na_mask = np.isin(
        subject_atom_array.chain_type,
        (ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID)
    )
    subject_c1_atom_array = subject_atom_array[
        subject_na_mask & (subject_atom_array.atom_name == "C1'")
    ]

    if len(reference_c1_atom_array) == 0 or len(subject_c1_atom_array) == 0:
        raise ValueError("No nucleic acid C1' atoms found for LDDT/GDDT.")
    if len(reference_c1_atom_array) != len(subject_c1_atom_array):
        raise ValueError(
            f"NA C1' atom count mismatch: reference={len(reference_c1_atom_array)}, "
            f"subject={len(subject_c1_atom_array)}."
        )

    c1_prime_lddt = biotite.structure.lddt(
        reference_c1_atom_array,
        subject_c1_atom_array
    )
    c1_prime_gddt = biotite.structure.lddt(
        reference_c1_atom_array,
        subject_c1_atom_array,
        inclusion_radius = 10000,
        distance_bins = (1.0, 2.0, 4.0, 8.0)
    )

    result = {
        "c1_prime_lddt": float(c1_prime_lddt),
        "c1_prime_gddt": float(c1_prime_gddt)
    }

    return result

def calculate_protein_aligned_na_c1_rmsd(reference_atom_array,
                                         subject_atom_array):
    """
    Given a reference and subject atom array (both containing protein and
    nucleic acid chains), superimposes the structures using protein CA atoms
    and then calculates the RMSD on nucleic acid C1' atoms.

    Args:
        reference_atom_array (AtomArray): The reference atom array containing
            both protein and nucleic acid chains.
        subject_atom_array (AtomArray): The subject (predicted) atom array
            containing both protein and nucleic acid chains.

    Returns:
        protein_aligned_na_rmsd (float): The C1' RMSD after superimposition
            on protein CA atoms.
    """
    # Extract protein CA atoms from reference.
    reference_protein_mask = np.isin(
        reference_atom_array.chain_type, (ChainType.POLYPEPTIDE_L,)
    )
    reference_ca_mask = reference_protein_mask & (reference_atom_array.atom_name == "CA")
    reference_ca = reference_atom_array[reference_ca_mask]

    # Extract protein CA atoms from subject.
    subject_protein_mask = np.isin(
        subject_atom_array.chain_type, (ChainType.POLYPEPTIDE_L,)
    )
    subject_ca_mask = subject_protein_mask & (subject_atom_array.atom_name == "CA")
    subject_ca = subject_atom_array[subject_ca_mask]

    # Extract NA C1' atoms from reference.
    reference_na_mask = np.isin(
        reference_atom_array.chain_type,
        (ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID)
    )
    reference_c1_mask = reference_na_mask & (reference_atom_array.atom_name == "C1'")
    reference_c1 = reference_atom_array[reference_c1_mask]

    # Extract NA C1' atoms from subject.
    subject_na_mask = np.isin(
        subject_atom_array.chain_type,
        (ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID)
    )
    subject_c1_mask = subject_na_mask & (subject_atom_array.atom_name == "C1'")
    subject_c1 = subject_atom_array[subject_c1_mask]

    if len(reference_ca) == 0 or len(subject_ca) == 0:
        raise ValueError("No protein CA atoms found for superimposition.")
    if len(reference_c1) == 0 or len(subject_c1) == 0:
        raise ValueError("No nucleic acid C1' atoms found for RMSD.")
    if len(reference_ca) != len(subject_ca):
        raise ValueError(
            f"Protein CA atom count mismatch: reference={len(reference_ca)}, "
            f"subject={len(subject_ca)}."
        )
    if len(reference_c1) != len(subject_c1):
        raise ValueError(
            f"NA C1' atom count mismatch: reference={len(reference_c1)}, "
            f"subject={len(subject_c1)}."
        )

    reference_protein_sequence = [
        NAConstants.protein_resname_to_one_letter.get(
            res_name,
            NAConstants.protein_unknown_restype
        )
        for res_name in reference_ca.res_name
    ]
    subject_protein_sequence = [
        NAConstants.protein_resname_to_one_letter.get(
            res_name,
            NAConstants.protein_unknown_restype
        )
        for res_name in subject_ca.res_name
    ]
    if reference_protein_sequence != subject_protein_sequence:
        raise ValueError(
            "Reference and subject protein sequences must match for "
            "protein-aligned nucleic acid RMSD."
        )

    # Concatenate protein CA and NA C1' atoms into combined arrays.
    reference_combined = reference_ca + reference_c1
    subject_combined = subject_ca + subject_c1

    # Create mask: True for protein CA atoms (used for alignment).
    atom_mask = (reference_combined.atom_name == "CA")

    # Superimpose using protein CA atoms; transformation applied to all atoms.
    superimposed, _ = biotite.structure.superimpose(
        reference_combined, subject_combined, atom_mask = atom_mask
    )

    # Extract the NA C1' portion from the superimposed result.
    superimposed_c1 = superimposed[superimposed.atom_name == "C1'"]

    # Compute RMSD on NA C1' only.
    protein_aligned_na_rmsd = float(
        biotite.structure.rmsd(
            reference_c1,
            superimposed_c1
        )
    )
    
    return protein_aligned_na_rmsd

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
                 het = 1,
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
                4: alignment of multiple monomeric chains into a consensus 
                    alignment.
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
            loaded_subject_structure_path (str): The path to the subject
                structure as loaded by US-Align.
            loaded_reference_structure_path (str): The path to the reference
                structure as loaded by US-Align.
            loaded_subject_length (int): The length of the subject structure as
                loaded by US-Align.
            loaded_reference_length (int): The length of the reference 
                structure as loaded by US-Align.
            aligned_length (int): The length of the aligned region between the
                subject and reference structures.
            sequence_identity (n_identical/n_aligned) (float): The sequence 
                identity (n_identical/n_aligned) between the subject and 
                reference structures in the aligned region.
            rmsd (float): The root mean square deviation (RMSD) between the 
                aligned structures.
            tm_score (float): The TM-score between the aligned structures,
                normalized by the length of the reference structure.
            error_sequence_too_short (bool): Whether the alignment failed due
                to the sequence being too short.
            error_cannot_parse_file (bool): Whether the alignment failed due to
                US-Align being unable to parse the input structure files.
    """
    # Check that the mol and atom options agree.
    if mol == "auto" and atom != "auto":
        raise ValueError(
            "If mol is set to 'auto', atom must also be set to 'auto'."
        )

    # Convert the structure paths to absolute paths.
    subject_structure_path = os.path.abspath(subject_structure_path)
    reference_structure_path = os.path.abspath(reference_structure_path)

    # Check that the structure paths exist.
    if not os.path.exists(subject_structure_path):
        raise ValueError(f"Structure file not found: {subject_structure_path}")
    if not os.path.exists(reference_structure_path):
        raise ValueError(
            f"Structure file not found: {reference_structure_path}"
        )

    # Create a temporary file for the US-Align output.
    us_align_output_file = tempfile.NamedTemporaryFile(mode = "wt")
    us_align_error_file = tempfile.NamedTemporaryFile(mode = "wt")

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
            stderr = us_align_error_file
        )

        us_align_output_text = read_text_file(us_align_output_file.name)
        us_align_error_text = read_text_file(us_align_error_file.name)
    
        # Extract the TM-score and RMSD from the US-Align output.
        loaded_subject_structure_path = None
        loaded_reference_structure_path = None
        loaded_subject_length = None
        loaded_reference_length = None
        aligned_length = None
        rmsd = None
        sequence_identity = None
        tm_score = None
        for line in us_align_output_text.split("\n"):
            if line.startswith("Name of Structure_1:"):
                loaded_subject_structure_path = line.split(
                    "Name of Structure_1:"
                )[1].split("(to be superimposed onto Structure_2)")[0].strip()
            elif line.startswith("Name of Structure_2:"):
                loaded_reference_structure_path = line.split(
                    "Name of Structure_2:"
                )[1].strip()
            elif line.startswith("Length of Structure_1:"):
                loaded_subject_length = int(line.split(
                    "Length of Structure_1:"
                )[1].split("residues")[0].strip())
            elif line.startswith("Length of Structure_2:"):
                loaded_reference_length = int(line.split(
                    "Length of Structure_2:"
                )[1].split("residues")[0].strip())
            elif line.startswith("Aligned length="):
                aligned_length = int(line.split(
                    "Aligned length="
                )[1].split(",")[0].strip())
                rmsd = float(line.split("RMSD=")[1].split(",")[0].strip())
                sequence_identity = float(line.split(
                    "Seq_ID=n_identical/n_aligned=")[1].strip()
                )
            elif (
                line.startswith("TM-score=") and 
                "normalized by length of Structure_2" in line
            ):
                tm_score = float(line.split(
                    "TM-score="
                )[1].split("(normalized by length of Structure_2")[0].strip())

        # Close files. 
        us_align_output_file.close()
        us_align_error_file.close()
        
        # Parse errors.
        error_sequence_too_short = False
        error_cannot_parse_file = False
        if tm_score is None:
            if "Sequence is too short" in us_align_error_text:
                error_sequence_too_short = True
            elif "Warning! Cannot parse file" in us_align_error_text:
                error_cannot_parse_file = True
            else:
                raise ValueError(
                    "Failed to extract RMSD and TM-score from US-Align output."
                )
        
        result = {
            "loaded_subject_structure_path": loaded_subject_structure_path,
            "loaded_reference_structure_path": loaded_reference_structure_path,
            "loaded_subject_length": loaded_subject_length,
            "loaded_reference_length": loaded_reference_length,
            "aligned_length": aligned_length,
            "sequence_identity (n_identical/n_aligned)": sequence_identity,
            "rmsd": rmsd,
            "tm_score": tm_score,
            "error_sequence_too_short": error_sequence_too_short,
            "error_cannot_parse_file": error_cannot_parse_file
        }

        return result
    except (subprocess.CalledProcessError, ValueError) as e:
        us_align_output_file.close()
        us_align_error_file.close()
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
    
def calculate_alignment_weight(ppm, eps = 1e-10):
    """
    Calculate the per-position alignment weight used for PPM alignment
    scoring.

    Arguments:
        ppm (np.float64 np.ndarray): an L x 4 array of representing the PPM.
        eps (float, optional): a small epsilon, to be added to each 
            probability to prevent taking a logarithm of 0.
    
    Returns:
        per_position_alignment_weight (np.float64 np.ndarray): the
            per-position alignment weight of the ppm.
    """
    # Check the shape of the ppm.
    assert(ppm.shape[-1] == 4)

    # Add epsilon to each column of probabilities and normalize.
    ppm_plus_eps = ppm + eps
    ppm_eps_norm = ppm_plus_eps / np.sum(ppm_plus_eps, axis = -1)[:, None]

    # Compute the per-position alignment weight.
    per_position_alignment_weight = np.sum(np.log(ppm_eps_norm) / np.log(0.25), axis = -1)

    return per_position_alignment_weight

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
    Calculate the per-position pearson correlation coefficient between the ppm
    and the one-hot nucleic acid sequence, weight each position by the
    alignment weight, and return the sum.

    Arguments:
        ppm (np.float64 np.ndarray): an L x 4 array of representing the PPM.
        S_one_hot (np.float64 np.ndarray): an L x 4 array representing the
            one-hot nucleic acid sequence.
    
    Returns:
        alignment_weighted_pcc_sum (float): the sum of the per-position
            pearson correlation coefficients after weighting by the alignment
            weights.
    """
    # Check the shape of the ppm and the S_one_hot.
    assert(ppm.shape[-1] == 4)
    assert(S_one_hot.shape[-1] == 4)

    # Weight each per-position pearson correlation coefficient by the alignment
    # weight.
    per_position_alignment_weight = calculate_alignment_weight(ppm)
    per_position_pcc = calculate_pearson_correlation_coeffcient(ppm, S_one_hot)
    per_position_alignment_weighted_pcc = per_position_pcc * (0.5 * per_position_alignment_weight)

    # Calculate the sum of the per_position_alignment_weighted_pcc.
    alignment_weighted_pcc_sum = np.sum(per_position_alignment_weighted_pcc)

    return alignment_weighted_pcc_sum

def weighted_align(ppm, S_one_hot_na, S_non_x_mask):
    """
    Given a ppm and a sequence, find the maximum alignment-weighted pearson
    correlation coefficient alignment(s).

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
                    # not DX at that position), and higher alignment weight
                    # otherwise.
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
                            ppm_col_alignment_weight = calculate_alignment_weight(ppm[ppm_idx][None, :])
                            aligned_ppm_col_alignment_weight = calculate_alignment_weight(aligned_ppm[S_idx, na_restype_ints_to_compare][None, :])
                            if ppm_col_alignment_weight > aligned_ppm_col_alignment_weight:
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
        tmp_directory = None
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
        if tmp_directory is not None:
            tmp_directory.cleanup()

        return design_data
    except (subprocess.CalledProcessError, ValueError) as e:
        # Clean up the temporary directory if it was created.
        if tmp_directory is not None:
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
        tmp_directory = None
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
        if tmp_directory is not None:
            tmp_directory.cleanup()

        return design_data
    except (subprocess.CalledProcessError, ValueError) as e:
        # Clean up the temporary directory if it was created.
        if tmp_directory is not None:
            tmp_directory.cleanup()
        raise e

def run_ridiffusion(structure_path,
                    output_directory = None,
                    n_samples = 1,
                    ridiffusion_apptainer_path = "/software/containers/users/akubaney/ridiffusion.sif",
                    ridiffusion_path = "/home/akubaney/software/RIdiffusion/seq_generator.py"):
    """
    Given a structure path, runs the RIdiffusion sequence design algorithm to
    generate sequences for the structure. The output is a list of dictionaries
    containing the design ID, name, design sequence, and tool-reported sequence
    recovery.

    Args:
        structure_path (str): The path to the structure file.
        output_directory (str): The path to the output directory. If not
            specified, a temporary directory will be created.
        n_samples (int): The number of samples to generate.
        ridiffusion_apptainer_path (str): The path to the RIdiffusion
            apptainer.
        ridiffusion_path (str): The path to the RIdiffusion run file.
    
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
        tmp_directory = None
        output_directory = os.path.abspath(output_directory)
    
    # Compute the output directory for the sequences.
    seqs_output_directory = os.path.join(output_directory, "seqs")

    # Create the output directory if it does not exist.
    os.makedirs(seqs_output_directory, exist_ok = True)

    # Compute the name of the structure.
    structure_name = os.path.splitext(os.path.basename(structure_path))[0]

    fasta_path = os.path.join(seqs_output_directory, f"{structure_name}.fa")

    # Run the RIdiffusion sequence design algorithm.
    input_pdb_directory = tempfile.TemporaryDirectory()
    try:
        input_structure_path = os.path.join(
            input_pdb_directory.name,
            os.path.basename(structure_path)
        )
        shutil.copy(structure_path, input_structure_path)

        subprocess.run(
            [
                "apptainer",
                "exec",
                "--nv",
                ridiffusion_apptainer_path,
                "python",
                ridiffusion_path,
                "--pdb_dir",
                str(input_pdb_directory.name),
                "--num_samples",
                str(n_samples),
                "--output_file",
                str(fasta_path)
            ],
            check = True,
            cwd = os.path.dirname(ridiffusion_path),
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        # Check that the output fasta file exists.
        if not os.path.exists(fasta_path):
            raise ValueError(f"Output fasta file not found: {fasta_path}")

        # Read the output fasta file.
        fasta_entries = read_fasta_file(fasta_path)

        # Skip the first entry of the fasta, which contains the parent sequence.
        fasta_entries = fasta_entries[1:]

        design_data = []
        for fasta_header, fasta_sequence in fasta_entries:
            fasta_header = fasta_header.strip()
            fasta_sequence = fasta_sequence.replace("\n", "")

            fasta_header_name, tool_reported_sequence_recovery = \
                fasta_header.split("--")
            design_id = fasta_header_name.split("_")[0].replace("seq", "")

            design_dict = {
                "input_structure_name": structure_name,
                "input_structure_path": structure_path,
                "design_id": design_id,
                "name": f"{structure_name}_{design_id}",
                "design_sequence": fasta_sequence,
                "tool_reported_sequence_recovery": float(tool_reported_sequence_recovery),
                "design_method": "ridiffusion",
                "model_weights_path": ""
            }

            design_data.append(design_dict)

        input_pdb_directory.cleanup()

        # Clean up the temporary directory if it was created.
        if tmp_directory is not None:
            tmp_directory.cleanup()

        return design_data
    except (subprocess.CalledProcessError, ValueError) as e:
        input_pdb_directory.cleanup()

        # Clean up the temporary directory if it was created.
        if tmp_directory is not None:
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
        tmp_directory = None
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
        if tmp_directory is not None:
            tmp_directory.cleanup()

        return design_data
    except (subprocess.CalledProcessError, ValueError) as e:
        # Clean up the temporary directory if it was created.
        if tmp_directory is not None:
            tmp_directory.cleanup()
        
        # Clean up the output file and output directory for the ith sample.
        output_directory_i.cleanup()
        output_file_i.close()

        raise e

def extract_na_sequence_data_from_design_sequence(design_sequence,
                                                  design_method):
    """
    Given a design sequence, extracts the designed nucleic acid sequences.

    Args:
        design_sequence (str): The raw design sequence returned by the design
            method.
        design_method (str): The design method used.

    Returns:
        na_sequence_data ((str, ChainType) list): The designed nucleic acid
            sequence data.
    """
    def infer_na_chain_type_from_na_mpnn_sequence(chain_sequence):
        has_dna_char = False
        has_rna_char = False
        has_protein_char = False

        for c in chain_sequence:
            if c in NAConstants.na_mpnn_dna_chars:
                has_dna_char = True
            elif c in NAConstants.na_mpnn_rna_chars:
                has_rna_char = True
            elif c in NAConstants.na_mpnn_protein_chars:
                has_protein_char = True
            else:
                raise ValueError(
                    f"Unable to classify design chain sequence: {chain_sequence}"
                )

        # Determine chain type.
        if not has_dna_char and not has_rna_char and has_protein_char:
            return None
        elif has_dna_char and not has_rna_char and not has_protein_char:
            return ChainType.DNA
        elif not has_dna_char and has_rna_char and not has_protein_char:
            return ChainType.RNA
        elif has_dna_char and has_rna_char and not has_protein_char:
            return ChainType.DNA_RNA_HYBRID
        elif has_protein_char:
            raise ValueError(
                "Unable to classify mixed protein/nucleic acid design chain "
                f"sequence: {chain_sequence}"
            )

        raise ValueError(
            f"Unable to classify empty design chain sequence: {chain_sequence}"
        )

    chain_sequences = design_sequence.split(NAConstants.chain_break_character)

    if design_method == "na_mpnn":
        na_sequence_data = []
        for chain_sequence in chain_sequences:
            if len(chain_sequence) == 0:
                raise Exception(
                    "Design sequence contains an empty chain sequence"
                )

            chain_type = infer_na_chain_type_from_na_mpnn_sequence(
                chain_sequence
            )
            if chain_type is None:
                continue

            na_sequence_data.append((chain_sequence, chain_type))
        
        na_sequence_data = standardize_na_sequence(
            na_sequence_data,
            method = "na_mpnn"
        )
    elif design_method in ("grnade", "ridiffusion", "rhodesign"):
        na_sequence_data = []
        for chain_sequence in chain_sequences:
            if len(chain_sequence) == 0:
                raise Exception(
                    "Design sequence contains an empty chain sequence"
                )
            
            na_sequence_data.append((chain_sequence, ChainType.RNA))
        
        na_sequence_data = standardize_na_sequence(na_sequence_data)
    else:
        raise ValueError(f"Invalid sequence design method: {design_method}")

    # Designed NA sequences must be concrete; downstream design processing does
    # not treat unknown residues as valid outputs.
    check_na_sequence_validity(
        na_sequence_data,
        unknown_residue_allowed = False
    )

    return na_sequence_data

################################################################################
# Combined Functionality
################################################################################

def find_best_reference_overlap(reference_na_sequence_data,
                                subject_na_sequence_data,
                                reference_atom_array,
                                subject_atom_array,
                                use_protein_alignment = False):
    """
    Given single-chain reference and subject nucleic acid sequence data where
    the subject is shorter, finds the best-overlapping reference subsequence.

    Args:
        reference_na_sequence_data ((str, ChainType) list): The reference
            nucleic acid sequence data.
        subject_na_sequence_data ((str, ChainType) list): The subject nucleic
            acid sequence data.
        reference_atom_array (AtomArray): The reference atom array. For
            protein-aligned mode, this should contain both protein and nucleic
            acid atoms. Otherwise the nucleic acid atoms are used and C1'
            atoms are selected internally for RMSD scoring.
        subject_atom_array (AtomArray): The subject atom array, following the
            same convention as reference_atom_array.
        use_protein_alignment (bool): Whether to score overlaps using
            protein-aligned NA RMSD. Default is False.

    Returns:
        result (dict): A dictionary containing:
            best_start_idx (int): The starting residue index of the best
                overlap in the reference sequence.
            best_end_idx (int): The exclusive ending residue index of the best
                overlap in the reference sequence.
            best_rmsd (float): The RMSD of the best overlap.
            reference_atom_array (AtomArray): The reference atom array trimmed
                to the best overlap.
            reference_na_sequence_data ((str, ChainType) list): The trimmed
                reference nucleic acid sequence data for scoring.
    """
    if len(reference_na_sequence_data) != 1 or len(subject_na_sequence_data) != 1:
        raise ValueError(
            "Best-overlap alignment only supports single-chain nucleic acid "
            "sequence data."
        )

    reference_sequence, reference_chain_type = reference_na_sequence_data[0]
    subject_sequence, subject_chain_type = subject_na_sequence_data[0]

    if subject_chain_type != reference_chain_type:
        raise ValueError(
            "Subject nucleic acid chain type must match the reference chain "
            "type for best-overlap alignment."
        )

    reference_sequence_length = len(reference_sequence)
    subject_sequence_length = len(subject_sequence)
    if subject_sequence_length >= reference_sequence_length:
        raise ValueError(
            "Best-overlap alignment requires the subject sequence to be "
            "shorter than the reference sequence."
        )

    reference_na_atom_array = reference_atom_array[
        np.isin(
            reference_atom_array.chain_type,
            (ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID)
        )
    ]
    subject_na_atom_array = subject_atom_array[
        np.isin(
            subject_atom_array.chain_type,
            (ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID)
        )
    ]
    reference_na_token_starts = get_token_starts(reference_na_atom_array)
    subject_na_token_starts = get_token_starts(subject_na_atom_array)

    if len(reference_na_token_starts) != reference_sequence_length or \
       len(subject_na_token_starts) != subject_sequence_length:
        raise ValueError(
            "Unable to align shorter nucleic acid sequences because the "
            "nucleic acid residue counts do not match the sequence "
            "lengths."
        )

    reference_na_token_ends = (
        list(reference_na_token_starts[1:]) +
        [len(reference_na_atom_array)]
    )

    if use_protein_alignment:
        reference_protein_atom_array = reference_atom_array[
            reference_atom_array.chain_type == ChainType.POLYPEPTIDE_L
        ]

    best_rmsd = None
    best_start_idx = None
    best_reference_atom_array = None
    for possible_start_idx in range(
        reference_sequence_length - subject_sequence_length + 1
    ):
        possible_end_idx = possible_start_idx + subject_sequence_length
        reference_start_idx = reference_na_token_starts[possible_start_idx]
        reference_end_idx = reference_na_token_ends[possible_end_idx - 1]

        if use_protein_alignment:
            candidate_reference_atom_array = (
                reference_protein_atom_array +
                reference_na_atom_array[reference_start_idx : reference_end_idx]
            )
            candidate_rmsd = calculate_protein_aligned_na_c1_rmsd(
                candidate_reference_atom_array,
                subject_atom_array
            )
        else:
            candidate_reference_atom_array = reference_na_atom_array[
                reference_start_idx : reference_end_idx
            ]
            candidate_rmsd = calculate_na_c1_rmsd(
                candidate_reference_atom_array,
                subject_atom_array
            )

        if best_rmsd is None or candidate_rmsd < best_rmsd:
            best_rmsd = candidate_rmsd
            best_start_idx = possible_start_idx
            best_reference_atom_array = candidate_reference_atom_array

    best_end_idx = best_start_idx + subject_sequence_length
    result = {
        "best_start_idx": best_start_idx,
        "best_end_idx": best_end_idx,
        "best_rmsd": float(best_rmsd),
        "reference_atom_array": best_reference_atom_array,
        "reference_na_sequence_data": [
            (
                reference_sequence[best_start_idx : best_end_idx],
                reference_chain_type
            )
        ],
    }
    
    return result

def trim_reference_dssr_output_to_overlap(reference_dssr_output,
                                          best_start_idx,
                                          best_end_idx):
    """
    Retained helper for legacy best-overlap scoring of monomer RNA designs.
    This is currently unused.

    Given a DSSR output dictionary and overlap bounds on the reference
    sequence, trims the sequence and secondary structure to the selected
    overlap. Base pairs that extend outside the overlap are first converted to
    loops before trimming.

    Args:
        reference_dssr_output (dict): DSSR output containing "sequence" and
            "secondary_structure".
        best_start_idx (int): Inclusive starting residue index of the overlap.
        best_end_idx (int): Exclusive ending residue index of the overlap.

    Returns:
        trimmed_reference_dssr_output (dict): A shallow copy of the DSSR output
            with the sequence and secondary structure trimmed to the overlap.
    """
    trimmed_reference_dssr_output = dict(reference_dssr_output)
    trimmed_reference_dssr_output["sequence"] = \
        trimmed_reference_dssr_output["sequence"][best_start_idx:best_end_idx]

    base_pair_indices, _ = \
        calculate_base_pairs_and_loops_from_secondary_structure(
            reference_dssr_output["secondary_structure"]
        )
    updated_secondary_structure = reference_dssr_output["secondary_structure"]
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

    trimmed_reference_dssr_output["secondary_structure"] = \
        updated_secondary_structure[best_start_idx:best_end_idx]

    return trimmed_reference_dssr_output

def design_nucleic_acid_sequence(structure_path,
                                 overall_output_directory,
                                 num_samples,
                                 temperature,
                                 method = "na_mpnn",
                                 na_mpnn_model_path = None,
                                 with_protein = True):
    """
    Given a structure path, an overall output directory, the number of samples,
    the temperature, and the sequence design method, runs the specified
    sequence design method to generate sequences for the structure. A JSON is
    created for each design, containing the design ID, name, designed nucleic
    acid sequence data, protein sequences (from native), and metadata.

    Only NA-MPNN supports DNA design and design in protein context.

    Args:
        structure_path (str): The path to the structure file.
        overall_output_directory (str): The path to the overall output directory.
        num_samples (int): The number of samples to generate.
        temperature (float): The temperature for the sequence design algorithm.
        method (str): The sequence design method to use. Options are "na_mpnn",
            "grnade", "ridiffusion", and "rhodesign". Default is "na_mpnn".
        na_mpnn_model_path (str): The path to the NA-MPNN model file. Required
            if method is "na_mpnn".
        with_protein (bool): Whether to include protein chains as structural
            context during design. If False and the structure contains protein,
            protein chains are removed before design. Default is True.

    Side Effects:
        Creates an output directory for the structure, copies the structure to
            the output directory, creates a subdirectory for the design JSON
            files, and saves a JSON file for each design.
    """
    # Convert the structure path and overall output directory to absolute paths.
    structure_path = os.path.abspath(structure_path)
    overall_output_directory = os.path.abspath(overall_output_directory)

    if method == "ridiffusion":
        temperature = None
    elif temperature is None:
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

    # Extract normalized sequence information from the native structure.
    reference_na_sequence_data, reference_protein_sequences = \
        extract_sequences_from_structure(structure_path)
    complex_sequence_data = prepare_complex_sequence_data(
        na_sequence_data = reference_na_sequence_data,
        protein_sequences = reference_protein_sequences
    )
    has_protein = complex_sequence_data["has_protein"]
    has_dna = complex_sequence_data["has_dna"]

    # Check method compatibility with DNA and protein context.
    if method != "na_mpnn" and has_dna:
        raise ValueError(
            f"Method '{method}' does not support DNA design. "
            f"Structure '{structure_name}' contains DNA chains."
        )
    if method != "na_mpnn" and has_protein and with_protein:
        raise ValueError(
            f"Method '{method}' does not support design in protein context."
        )

    # Rewrite non-canonical NA residues to their canonical equivalents so
    # downstream design tools can process the structure.
    canonicalized_structure_path, temp_canonicalized_directory = \
        canonicalize_noncanonical_na_residues(structure_path)

    # If with_protein is False and the structure has protein, create a
    # temporary PDB with protein chains removed.
    temp_na_only_path = None
    temp_na_only_directory = None
    design_structure_path = canonicalized_structure_path
    if not with_protein and has_protein:
        temp_na_only_path, temp_na_only_directory = \
            remove_protein_chains_from_structure(
                canonicalized_structure_path
            )
        design_structure_path = temp_na_only_path

    # Design JSON output directory.
    design_json_output_directory = os.path.join(output_directory, "design_json")
    os.makedirs(design_json_output_directory)

    try:
        if method == "na_mpnn":
            # Run NA-MPNN sequence design.
            design_data = run_na_mpnn_sequence(
                design_structure_path,
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
                design_structure_path,
                output_directory = output_directory,
                n_samples = num_samples,
                temperature = temperature
            )
        elif method == "ridiffusion":
            # Run RIdiffusion sequence design.
            design_data = run_ridiffusion(
                design_structure_path,
                output_directory = output_directory,
                n_samples = num_samples
            )
        elif method == "rhodesign":
            # Run RhoDesign sequence design.
            design_data = run_rhodesign(
                design_structure_path,
                output_directory = output_directory,
                n_samples = num_samples,
                temperature = temperature
            )
        else:
            raise ValueError(f"Invalid sequence design method: {method}")
    finally:
        # Clean up the temporary NA-only structure if it was created.
        if temp_na_only_directory is not None:
            temp_na_only_directory.cleanup()
        if temp_canonicalized_directory is not None:
            temp_canonicalized_directory.cleanup()

    # Write the design data to a JSON file.
    for design_dict in design_data:
        design_dict["original_input_structure_path"] = original_structure_path
        design_dict["input_structure_name"] = structure_name
        design_dict["input_structure_path"] = structure_path
        design_dict["name"] = f"{structure_name}_{design_dict['design_id']}"
        design_dict["na_sequence_data"] = extract_na_sequence_data_from_design_sequence(
            design_dict["design_sequence"],
            design_dict["design_method"]
        )
        design_dict["protein_sequences"] = reference_protein_sequences
        design_dict["with_protein"] = with_protein

        design_json_path = os.path.join(
            design_json_output_directory,
            f"{design_dict['name']}.json"
        )
        write_json_file(design_json_path, design_dict)

def process_reference(reference_structure_path,
                      overall_output_directory):
    """
    Given a reference structure path and an overall output directory,
    processes the reference structure by extracting normalized sequence data
    using AtomWorks, extracting secondary structure with DSSR (for monomer
    RNA only), and running the AlphaFold3 data pipeline for protein chains
    (to pre-compute MSAs and templates).

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

    # Check that the reference structure path exists.
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

    # Extract normalized sequence data from the structure.
    na_sequence_data, protein_sequences = extract_sequences_from_structure(
        reference_structure_path
    )
    complex_sequence_data = prepare_complex_sequence_data(
        na_sequence_data = na_sequence_data,
        protein_sequences = protein_sequences
    )
    has_protein = complex_sequence_data["has_protein"]
    is_monomer_rna = complex_sequence_data["is_monomer_rna"]

    # Build the output dictionary.
    output_dict = {
        "name": structure_name,
        "original_reference_structure_path": original_reference_structure_path,
        "reference_structure_path": reference_structure_path,
        "na_sequence_data": na_sequence_data,
        "protein_sequences": protein_sequences,
    }

    # For monomer RNA, run DSSR to extract sequence and secondary structure.
    if is_monomer_rna:
        # Run dssr.
        dssr_output = run_dssr(reference_structure_path)
        reference_rna_sequence = na_sequence_data[0][0]

        # Standardize the dssr sequence.
        dssr_chain_type = na_sequence_data[0][1]
        dssr_sequence_data = standardize_na_sequence(
            [(dssr_output["sequence"], dssr_chain_type)],
            method = "dssr"
        )
        dssr_output["sequence"] = dssr_sequence_data[0][0]

        # Check that sequence is valid.
        check_na_sequence_validity(
            dssr_sequence_data,
            unknown_residue_allowed = True
        )

        # Standardize the dssr secondary structure.
        dssr_output["secondary_structure"] = \
            standardize_secondary_structure(
                dssr_output["secondary_structure"],
                method = "dssr"
            )

        if len(dssr_output["sequence"]) != len(reference_rna_sequence):
            raise ValueError(
                "Reference DSSR sequence length must match the reference RNA "
                "sequence length for monomer RNA processing."
            )
        if len(dssr_output["secondary_structure"]) != len(reference_rna_sequence):
            raise ValueError(
                "Reference DSSR secondary structure length must match the "
                "reference RNA sequence length for monomer RNA processing."
            )

        output_dict["dssr"] = dssr_output

    # For structures with protein, run the AlphaFold3 data pipeline for the
    # full protein complex to pre-compute MSAs and templates.
    if has_protein:
        protein_data_pipeline_directory = os.path.abspath(
            os.path.join(
                output_directory,
                "af3_protein_data_pipeline"
            )
        )
        os.makedirs(protein_data_pipeline_directory, exist_ok = True)

        # Use a stable hash of the full protein complex sequence list for the
        # directory name to avoid filesystem path length issues.
        protein_complex_hash = stable_sequence_hash(
            "||".join(protein_sequences)
        )
        pipeline_name = f"{structure_name}_{protein_complex_hash}"

        # Run AlphaFold3 data pipeline only (no inference).
        af3_result = run_alphafold3(
            name = pipeline_name,
            chain_sequence_data = [
                (protein_seq, ChainType.POLYPEPTIDE_L)
                for protein_seq in protein_sequences
            ],
            output_dir = protein_data_pipeline_directory,
            run_data_pipeline = True,
            run_inference = False,
            num_seeds = 1
        )

        data_json = read_json_file(af3_result["json_input_path"])
        af3_protein_chain_data = {}
        for protein_sequence in protein_sequences:
            found_match = False
            for sequence_entry in data_json["sequences"]:
                _sequence_type, seq_data = next(iter(sequence_entry.items()))
                if seq_data["sequence"] != protein_sequence:
                    continue

                af3_protein_chain_data[protein_sequence] = {
                    "unpairedMsa": seq_data["unpairedMsa"],
                    "pairedMsa": seq_data["pairedMsa"],
                    "templates": seq_data["templates"]
                }
                found_match = True
                break

            if not found_match:
                raise ValueError(
                    "AlphaFold3 protein data pipeline output is missing MSA/"
                    "template data for a protein sequence in the reference "
                    "structure."
                )

        output_dict["af3_protein_chain_data"] = af3_protein_chain_data

    # Save the output dictionary to a JSON file.
    output_json_path = os.path.join(reference_json_output_directory,
                                    f"{structure_name}.json")
    write_json_file(output_json_path, output_dict)

def process_design(subject_path,
                   overall_output_directory,
                   reference_path = None):
    """
    Given a design path and an overall output directory, processes the design
    by running appropriate prediction tools based on the complex type:
      - Monomer RNA without protein: EternaFold, RibonanzaNet, AlphaFold3.
      - Other NA without protein: AlphaFold3 only.
      - With protein: AlphaFold3 with pre-computed protein MSAs/templates.

    If the native example contains protein chains, the design is always folded
    in protein context using the native protein sequences. The with_protein
    flag only affects the design stage.
    The results are saved to a JSON file.

    Args:
        subject_path (str): The path to the design JSON file.
        overall_output_directory (str): The path to the overall output
            directory.
        reference_path (str): The path to the reference output JSON. Required
            when the structure has protein. Default is None.

    Side Effects:
        Creates an output directory for the design and saves a JSON file
            with the results of the predictions.
    """
    # Convert the subject path and overall output directory to absolute paths.
    subject_path = os.path.abspath(subject_path)
    overall_output_directory = os.path.abspath(overall_output_directory)

    # Check that the subject path exists.
    if not os.path.exists(subject_path):
        raise ValueError(f"Design JSON file not found: {subject_path}")

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

    # Extract design metadata.
    na_sequence_data = design_json["na_sequence_data"]
    protein_sequences = design_json["protein_sequences"]
    complex_sequence_data = prepare_complex_sequence_data(
        na_sequence_data = na_sequence_data,
        protein_sequences = protein_sequences
    )
    with_protein = design_json["with_protein"]
    has_protein = complex_sequence_data["has_protein"]
    is_monomer_rna = complex_sequence_data["is_monomer_rna"]

    # Build the output dictionary.
    output_dict = {
        "name": design_name,
        "na_sequence_data": na_sequence_data,
        "protein_sequences": protein_sequences,
        "with_protein": with_protein,
        "design_input_path": subject_path,
    }

    alphafold3_chain_sequence_data = list(na_sequence_data)
    precomputed_chain_data = None

    # Case A/B: NA-only, no protein.
    if not has_protein:
        if is_monomer_rna:
            # Get the design sequence.
            design_sequence = na_sequence_data[0][0]

            # Predict the secondary structure of the design sequence with
            # EternaFold.
            eternafold_result = run_eternafold(design_sequence)
            output_dict["eternafold"] = eternafold_result

            # Predict the secondary structure and reactivity profile of the
            # design sequence with RiboNanzaNet.
            ribonanza_net_secondary_structure_result = \
                run_ribonanza_net_secondary_structure(design_sequence)
            ribonanza_net_reactivity_profile_result = \
                run_ribonanza_net_reactivity_profile(design_sequence)
            output_dict["ribonanza_net_secondary_structure"] = \
                ribonanza_net_secondary_structure_result
            output_dict["ribonanza_net_reactivity_profile"] = \
                ribonanza_net_reactivity_profile_result

    # Case C: With protein.
    else:
        if reference_path is None:
            raise ValueError(
                "reference_path is required when structure has protein."
            )

        reference_path = os.path.abspath(reference_path)
        if not os.path.exists(reference_path):
            raise ValueError(f"Reference JSON file not found: {reference_path}")

        # Load the reference JSON to get pre-computed protein AF3 data.
        reference_json = read_json_file(reference_path)
        precomputed_chain_data = reference_json["af3_protein_chain_data"]

        # Build sequences list: NA chains first, then protein chains.
        for protein_sequence in protein_sequences:
            alphafold3_chain_sequence_data.append(
                (protein_sequence, ChainType.POLYPEPTIDE_L)
            )

    alphafold3_result = run_alphafold3(
        name = design_name,
        chain_sequence_data = alphafold3_chain_sequence_data,
        output_dir = output_directory,
        num_diffusion_samples = 5,
        num_seeds = 1,
        run_data_pipeline = False,
        precomputed_chain_data = precomputed_chain_data,
    )
    output_dict["alphafold3"] = alphafold3_result

    # Save the output dictionary to a JSON file.
    output_json_path = os.path.join(processed_design_json_output_directory,
                                    f"{design_name}.json")
    write_json_file(output_json_path, output_dict)

def score_design(reference_path,
                 subject_path,
                 overall_output_directory):
    """
    Given a reference path and a subject path, scores the design by comparing
    the reference and subject sequences, secondary structures, reactivity
    profiles, and/or structures depending on the complex type.

    Scoring modes:
      - Monomer RNA without protein: sequence recovery, GC content, secondary
        structure F1 scores, reactivity profile scores, C1' RMSD/lDDT/gDDT,
        AF3 confidence metrics.
      - Other NA without protein: sequence recovery, GC content, C1'
        RMSD/lDDT/gDDT, AF3 confidence metrics.
      - With protein: sequence recovery, GC content, protein-aligned NA C1'
        RMSD, iPTM, MinPAE, AF3 confidence metrics.

    Args:
        reference_path (str): The path to the reference output JSON.
        subject_path (str): The path to the subject output json.
        overall_output_directory (str): The path to the overall output
            directory.

    Side Effects:
        Creates an output directory for the subject and saves a JSON file
            with the results of the scoring.
    """
    if reference_path is None:
        raise ValueError(
            "reference_path is required for score_design."
        )

    # Convert the reference path and subject path to absolute paths.
    reference_path = os.path.abspath(reference_path)
    subject_path = os.path.abspath(subject_path)

    # Check that the reference path exists.
    if not os.path.exists(reference_path):
        raise ValueError(f"Reference file not found: {reference_path}")

    # Check that the subject path exists.
    if not os.path.exists(subject_path):
        raise ValueError(f"Subject file not found: {subject_path}")
    
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

    # Determine the complex type.
    reference_na_sequence_data = reference_output["na_sequence_data"]
    reference_protein_sequences = reference_output["protein_sequences"]
    subject_na_sequence_data = subject_output["na_sequence_data"]
    subject_protein_sequences = subject_output["protein_sequences"]
    
    reference_complex_sequence_data = prepare_complex_sequence_data(
        na_sequence_data = reference_na_sequence_data,
        protein_sequences = reference_protein_sequences
    )
    subject_complex_sequence_data = prepare_complex_sequence_data(
        na_sequence_data = subject_na_sequence_data,
        protein_sequences = subject_protein_sequences
    )

    with_protein = subject_output["with_protein"]
    reference_has_protein = reference_complex_sequence_data["has_protein"]
    subject_has_protein = subject_complex_sequence_data["has_protein"]
    reference_is_single_rna_chain = reference_complex_sequence_data[
        "is_single_rna_chain"
    ]
    subject_is_single_rna_chain = subject_complex_sequence_data[
        "is_single_rna_chain"
    ]
    if reference_has_protein != subject_has_protein:
        raise ValueError(
            "Reference and subject must agree on whether protein chains are "
            "present."
        )
    if reference_is_single_rna_chain != subject_is_single_rna_chain:
        raise ValueError(
            "Reference and subject must agree on whether they are single "
            "RNA-chain complexes."
        )

    has_protein = reference_has_protein
    is_single_rna_chain = reference_is_single_rna_chain
    is_monomer_rna = is_single_rna_chain and not has_protein
    if has_protein and reference_protein_sequences != subject_protein_sequences:
        raise ValueError(
            "Reference and subject protein sequences must match for "
            "protein-context scoring."
        )

    # GC content.
    subject_gc = calculate_gc_content(subject_na_sequence_data)
    reference_gc = calculate_gc_content(reference_na_sequence_data)
    delta_gc = subject_gc - reference_gc

    reference_sequence_length = sum(
        len(sequence) for sequence, _ in reference_na_sequence_data
    )
    subject_sequence_length = sum(
        len(sequence) for sequence, _ in subject_na_sequence_data
    )
    if subject_sequence_length != reference_sequence_length:
        raise ValueError(
            "Subject and reference nucleic acid sequence lengths must match "
            "for design scoring. "
            f"Subject length: {subject_sequence_length}; "
            f"reference length: {reference_sequence_length}."
        )

    # Start building the output dictionary.
    output_dict = {
        "reference_name": reference_output["name"],
        "reference_path": reference_path,
        "reference_sequence_length": reference_sequence_length,
        "subject_name": subject_output["name"],
        "subject_path": subject_path,
        "subject_sequence_length": subject_sequence_length,
        "with_protein": with_protein,
        "gc_content": subject_gc,
        "reference_gc_content": reference_gc,
        "delta_gc_content": delta_gc,
        "alphafold3_ptm": subject_output["alphafold3"]["ptm"],
        "alphafold3_iptm": subject_output["alphafold3"]["iptm"],
        "alphafold3_plddt": subject_output["alphafold3"]["plddt"],
        "alphafold3_pae": subject_output["alphafold3"]["pae"],
        "alphafold3_chain_pair_pae_min": subject_output["alphafold3"][
            "chain_pair_pae_min"
        ],
        "alphafold3_min_cross_chain_pae": subject_output["alphafold3"][
            "min_cross_chain_pae"
        ],
    }

    # Load the full atom arrays once for downstream structural metrics.
    subject_atom_array = load_first_assembly_atom_array(
        subject_output["alphafold3"]["predicted_structure_path"],
        add_missing_atoms = False
    )
    reference_atom_array = load_first_assembly_atom_array(
        reference_output["reference_structure_path"],
        add_missing_atoms = False
    )

    # With-protein scoring is always used for native protein-containing
    # examples, even if the sequence design itself was run without protein
    # context.
    if has_protein:
        protein_aligned_na_c1_prime_rmsd = calculate_protein_aligned_na_c1_rmsd(
            reference_atom_array,
            subject_atom_array
        )
        output_dict["alphafold3_protein_aligned_na_c1_prime_rmsd"] = \
            protein_aligned_na_c1_prime_rmsd

    else:
        # Secondary structure and reactivity metrics (monomer RNA only).
        if is_monomer_rna:
            reference_dssr_output = dict(reference_output["dssr"])

            # Compare the reference secondary structure to the eternafold
            # predicted secondary structure.
            eternafold_secondary_structure_result = \
                calculate_secondary_structure_stats(
                    reference_dssr_output["secondary_structure"],
                    subject_output["eternafold"]["predicted_secondary_structure"]
                )
            output_dict["eternafold_f1_score_pairs"] = \
                eternafold_secondary_structure_result["f1_score_pairs"]
            output_dict["eternafold_f1_score_loops"] = \
                eternafold_secondary_structure_result["f1_score_loops"]

            # Compare the reference secondary structure to the ribonanza net
            # predicted secondary structures.
            ribonanza_net_secondary_structure_result = dict()
            for predicted_secondary_structure in subject_output[
                "ribonanza_net_secondary_structure"
            ]["predicted_secondary_structures"]:
                predicted_secondary_structure = standardize_secondary_structure(
                    predicted_secondary_structure,
                    method = "ribonanzanet"
                )
                individual_result = calculate_secondary_structure_stats(
                    reference_dssr_output["secondary_structure"],
                    predicted_secondary_structure
                )

                # Append the results for each ribonanza net predicted
                # secondary structure to the ribonanza net secondary
                # structure result.
                for metric_name, metric_value in individual_result.items():
                    if metric_name not in \
                       ribonanza_net_secondary_structure_result:
                        ribonanza_net_secondary_structure_result[
                            metric_name
                        ] = []
                    ribonanza_net_secondary_structure_result[
                        metric_name
                    ].append(metric_value)

            # Calculate the mean of the ribonanza net secondary structure
            # results.
            for metric_name, metric_values in list(
                ribonanza_net_secondary_structure_result.items()
            ):
                ribonanza_net_secondary_structure_result[
                    f"mean_{metric_name}"
                ] = np.mean(metric_values)

            output_dict["ribonanza_net_f1_score_pairs"] = \
                ribonanza_net_secondary_structure_result[
                    "mean_f1_score_pairs"
                ]
            output_dict["ribonanza_net_f1_score_loops"] = \
                ribonanza_net_secondary_structure_result[
                    "mean_f1_score_loops"
                ]

            # Compare the reference secondary structure to the ribonanza net
            # predicted reactivity profiles.
            ribonanza_net_reactivity_profile_result = dict()
            for predicted_reactivity_profile in subject_output[
                "ribonanza_net_reactivity_profile"
            ]["predicted_2A3_reactivity_profiles"]:
                individual_result = calculate_reactivity_profile_score(
                    reference_dssr_output["secondary_structure"],
                    predicted_reactivity_profile
                )

                # Append the results for each ribonanza net predicted
                # reactivity profile to the ribonanza net reactivity profile
                # result.
                for metric_name, metric_value in individual_result.items():
                    if metric_name not in \
                       ribonanza_net_reactivity_profile_result:
                        ribonanza_net_reactivity_profile_result[
                            metric_name
                        ] = []
                    ribonanza_net_reactivity_profile_result[
                        metric_name
                    ].append(metric_value)

            # Calculate the mean of the ribonanza net reactivity profile
            # results.
            for metric_name, metric_values in list(
                ribonanza_net_reactivity_profile_result.items()
            ):
                ribonanza_net_reactivity_profile_result[
                    f"mean_{metric_name}"
                ] = np.mean(metric_values)

            output_dict["ribonanza_net_eternafold_class_score"] = \
                ribonanza_net_reactivity_profile_result[
                    "mean_eternafold_class_score"
                ]
            output_dict["ribonanza_net_crossed_pair_quality_score"] = \
                ribonanza_net_reactivity_profile_result[
                    "mean_crossed_pair_quality_score"
                ]
            output_dict["ribonanza_net_openknot_score"] = \
                ribonanza_net_reactivity_profile_result[
                    "mean_openknot_score"
                ]

        c1_prime_rmsd = calculate_na_c1_rmsd(
            reference_atom_array,
            subject_atom_array
        )
        c1_prime_lddt_gddt_result = calculate_na_c1_lddt_gddt(
            reference_atom_array,
            subject_atom_array
        )

        output_dict["alphafold3_c1_prime_rmsd"] = float(c1_prime_rmsd)
        output_dict["alphafold3_c1_prime_lddt"] = \
            c1_prime_lddt_gddt_result["c1_prime_lddt"]
        output_dict["alphafold3_c1_prime_gddt"] = \
            c1_prime_lddt_gddt_result["c1_prime_gddt"]

    # Compare the reference and subject nucleic acid sequences.
    sequence_recovery_result = calculate_sequence_recovery(
        reference_na_sequence_data,
        subject_na_sequence_data,
        unknown_residue_allowed_in_reference = True
    )
    output_dict["sequence_recovery"] = sequence_recovery_result[
        "sequence_recovery"
    ]

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
            "na_mpnn", "deeppbs", "rclamps_zf", and "rclamps_homeodomain".
            Default is "na_mpnn".
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
    elif method in ("rclamps_zf", "rclamps_homeodomain"):
        if method == "rclamps_zf":
            domain_type = "zf-C2H2"
        elif method == "rclamps_homeodomain":
            domain_type = "homeodomain"
        else:
            raise ValueError(f"Invalid rCLAMPS domain type: {method}")
        
        # Run rCLAMPS specificity prediction.
        specificity_data = run_rclamps(
            structure_path,
            output_directory = output_directory,
            domain_type = domain_type
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

    # Convert numpy arrays to lists and numpy scalars to Python int/float for
    # JSON serialization.
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, np.generic):
            result[k] = v.item()
    
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
    argument_parser.add_argument(
        "--with_protein",
        type = int,
        help = "Whether to include protein context during design (0 or 1).",
        default = 1
    )

    # Parse the command line arguments.
    args = argument_parser.parse_args()

    if args.function_name == "design_nucleic_acid_sequence":
        design_nucleic_acid_sequence(args.structure_path,
                                     args.overall_output_directory,
                                     args.num_samples,
                                     args.temperature,
                                     method = args.method,
                                     na_mpnn_model_path = args.na_mpnn_model_path,
                                     with_protein = bool(args.with_protein))
    elif args.function_name == "process_reference":
        process_reference(
            args.reference_structure_path,
            args.overall_output_directory
        )
    elif args.function_name == "process_design":
        process_design(args.subject_path,
                       args.overall_output_directory,
                       reference_path = args.reference_path)
    elif args.function_name == "score_design":
        score_design(args.reference_path,
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
