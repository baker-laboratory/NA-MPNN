import numpy as np
import pandas as pd
import torch
import itertools
import copy
import ast

def sample_bernoulli_rv(p):
    """
    Given a probability p, representing the success probability of a Bernoulli
    distribution, sample X ~ Bernoulli(p).

    Arguments:
        p (float): a float between 0 and 1, representing the success probability
            of a Bernoulli distribution.
    
    Returns:
        x (int): the result of sampling the random variable X ~ Bernoulli(p).
            P(X = 1) = p
            P(X = 0) = 1 - p.
    """
    # Check that 0 <= p <= 1.
    if p < 0 or p > 1:
        raise ValueError("The success probability p must be between 0 and 1 inclusive.")
    
    # Handle the edge cases, otherwise utilize the numpy uniform distribution.
    if p == 0:
        x = 0
    elif p == 1:
        x = 1
    else:
        # Sample the Y ~ Uniform(0, 1) distribution.
        uniform_sample = np.random.uniform(0.0, 1.0)

        # P(Y < p) = p.
        if uniform_sample < p:
            x = 1
        else:
            x = 0
    
    return x

def sample_bernoulli_rvs(p, n):
    """
    Given a probability p, representing the success probability of a Bernoulli
    distribution, and a number of samples n, sample X ~ Bernoulli(p) n times
    independently.

    Arguments:
        p (float): a float between 0 and 1, representing the success probability
            of a Bernoulli distribution.
        n (int): the number of samples to draw.
    
    Returns:
        x (np.int32 np.ndarray): an n length array; the result of sampling the 
            random variable X ~ Bernoulli(p) n times.
            P(X = 1) = p
            P(X = 0) = 1 - p.
    """
    # Sample Bernoulli(p) distribution n times.
    x = []
    for i in range(n):
        x.append(sample_bernoulli_rv(p))
    
    # Convert to numpy array.
    x = np.array(x, dtype = np.int32)

    return x

class PDBDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 cif_parser,
                 pdb_parser,
                 atom_list_to_save=['N', 'CA', 'C', 'O', #protein atoms
                                    'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'" #nucleic acid atoms
                                   ],
                 parse_protein=1,
                 parse_dna=1,
                 parse_rna=1,
                 parse_rna_as_dna=0,
                 na_shared_tokens=0,
                 protein_backbone_occ_cutoff=0.8,
                 protein_side_chain_occ_cutoff=0.5, 
                 dna_backbone_occ_cutoff=0.8,
                 dna_side_chain_occ_cutoff=0.5,
                 rna_backbone_occ_cutoff=0.8,
                 rna_side_chain_occ_cutoff=0.5,
                 crop_large_structures=0,
                 batch_tokens=6000,
                 na_ref_atom="C1'",
                 parse_ppms=0,
                 min_overlap_length=5,
                 drop_protein_probability=0,
                 na_only_as_uniform_ppm=0,
                 protein_interface_residue_mutation_probability=0,
                 mutate_base_pair_together=0,
                 mutate_entire_side_chain_interface_probability=0,
                 na_non_interface_as_uniform_ppm=0):
        self.protein_backbone_occ_cutoff = protein_backbone_occ_cutoff
        self.protein_side_chain_occ_cutoff = protein_side_chain_occ_cutoff
        self.dna_backbone_occ_cutoff = dna_backbone_occ_cutoff
        self.dna_side_chain_occ_cutoff = dna_side_chain_occ_cutoff
        self.rna_backbone_occ_cutoff = rna_backbone_occ_cutoff
        self.rna_side_chain_occ_cutoff = rna_side_chain_occ_cutoff

        self.parse_protein = parse_protein
        self.parse_dna = parse_dna
        self.parse_rna = parse_rna
        self.parse_rna_as_dna = parse_rna_as_dna
        self.na_shared_tokens = na_shared_tokens

        self.crop_large_structures = crop_large_structures
        self.batch_tokens = batch_tokens
        self.na_ref_atom = na_ref_atom

        self.drop_protein_probability = drop_protein_probability
        self.na_only_as_uniform_ppm = na_only_as_uniform_ppm
        self.protein_interface_residue_mutation_probability = protein_interface_residue_mutation_probability
        self.mutate_base_pair_together = mutate_base_pair_together
        self.mutate_entire_side_chain_interface_probability = mutate_entire_side_chain_interface_probability
        self.na_non_interface_as_uniform_ppm = na_non_interface_as_uniform_ppm

        self.parse_ppms = parse_ppms
        self.min_overlap_length = min_overlap_length

        self.atom_list_to_save = atom_list_to_save

        self.num_atoms_to_save = len(self.atom_list_to_save)

        self.atom_dict = dict(zip(self.atom_list_to_save, range(self.num_atoms_to_save)))

        self.cif_parser = cif_parser
        self.pdb_parser = pdb_parser

        self.polytypes = [
            'PP',
            'DNA',
            'RNA',
            'UNK',
            'MAS',
            'PAD'
        ]

        self.polytype_to_int = dict(zip(self.polytypes, range(len(self.polytypes))))

        if self.parse_rna_as_dna:
            self.polytype_to_int["RNA"] = self.polytype_to_int["DNA"]

        self.restypes = [
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

        self.protein_restypes = [
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
            'UNK'
        ]

        self.dna_restypes = [
            'DA',
            'DC',
            'DG',
            'DT',
            'DX'
        ]

        self.rna_restypes = [
            'A',
            'C',
            'G',
            'U',
            'RX'
        ]

        self.unknown_restypes = [
            "UNK",
            "DX",
            "RX"
        ]

        self.num_protein_restypes = len(self.protein_restypes)
        self.num_dna_restypes = len(self.dna_restypes)
        self.num_rna_restypes = len(self.rna_restypes)

        self.restype_3_to_1 = {
            'ALA': 'A', 
            'ARG': 'R', 
            'ASN': 'N', 
            'ASP': 'D', 
            'CYS': 'C', 
            'GLN': 'Q', 
            'GLU': 'E', 
            'GLY': 'G', 
            'HIS': 'H', 
            'ILE': 'I', 
            'LEU': 'L', 
            'LYS': 'K', 
            'MET': 'M', 
            'PHE': 'F', 
            'PRO': 'P', 
            'SER': 'S', 
            'THR': 'T', 
            'TRP': 'W', 
            'TYR': 'Y', 
            'VAL': 'V',
            'UNK': 'X',
            'DA': 'a',
            'DC': 'c',
            'DG': 'g',
            'DT': 't',
            'DX': 'x',
            'A': 'b',
            'C': 'd',
            'G': 'h',
            'U': 'u',
            'RX': 'y',
            'MAS': '-',
            'PAD': '+'
        }

        self.restype_to_int = dict(zip(self.restypes, range(len(self.restypes))))
        self.int_to_restype = dict(zip(range(len(self.restypes)), self.restypes))

        if self.parse_rna_as_dna or self.na_shared_tokens:
            self.restype_to_int["A"] = self.restype_to_int["DA"]
            self.restype_to_int["C"] = self.restype_to_int["DC"]
            self.restype_to_int["G"] = self.restype_to_int["DG"]
            self.restype_to_int["U"] = self.restype_to_int["DT"]
            self.restype_to_int["RX"] = self.restype_to_int["DX"]
        
        self.protein_restype_ints = list(map(lambda x: self.restype_to_int[x], self.protein_restypes))
        self.dna_restype_ints = list(map(lambda x: self.restype_to_int[x], self.dna_restypes))
        self.rna_restype_ints = list(map(lambda x: self.restype_to_int[x], self.rna_restypes))
        self.unknown_restype_ints = list(map(lambda x: self.restype_to_int[x], self.unknown_restypes))

        self.na_canonical_base_pair_restypes = [
            ('DA', 'DT'), 
            ('DA', 'U'), 
            ('DC', 'DG'), 
            ('DC', 'G'), 
            ('DG', 'DC'), 
            ('DG', 'C'), 
            ('DT', 'DA'), 
            ('DT', 'A'), 
            ('A', 'DT'), 
            ('A', 'U'), 
            ('C', 'DG'), 
            ('C', 'G'), 
            ('G', 'DC'), 
            ('G', 'C'), 
            ('U', 'DA'), 
            ('U', 'A')
        ]

        self.na_canonical_base_pair_ints = []
        for (restype_i, restype_j) in self.na_canonical_base_pair_restypes:
            self.na_canonical_base_pair_ints.append((self.restype_to_int[restype_i], 
                                                     self.restype_to_int[restype_j]))

        self.protein_bb_idx_list = []
        self.dna_bb_idx_list = []
        self.rna_bb_idx_list = []

        self.protein_backbone_list = ["N", "CA", "C", "O"]
        self.dna_backbone_list = ['OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]
        self.rna_backbone_list = ['OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]

        for atom in self.atom_list_to_save:
            if atom in self.protein_backbone_list:
                self.protein_bb_idx_list.append(self.atom_dict[atom])

        for atom in self.atom_list_to_save:
            if atom in self.dna_backbone_list:
                self.dna_bb_idx_list.append(self.atom_dict[atom])
        
        for atom in self.atom_list_to_save:
            if atom in self.rna_backbone_list:
                self.rna_bb_idx_list.append(self.atom_dict[atom])

    def __getitem__(self, index):
        """
        index = [[(example_dict, assembly_id), (example_dict, assembly_id)]]
        """
        x = [self.loader(example_dict, assembly_id) for (example_dict, assembly_id) in index[0]]
        return x

    def parse_structure(self, structure_path):
        if structure_path[-4:] == ".pdb" or structure_path[-7:] == ".pdb.gz":
            return self.pdb_parser.parse(structure_path)
        elif structure_path[-4:] == ".cif" or structure_path[-7:] == ".cif.gz":
            return self.cif_parser.parse(structure_path)
        else:
            raise Exception(f"{structure_path}: Unknown structure path extension.")

    def load_ppms(self, ppm_paths_str, randomize_experimental_ppms):
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
    
    def calculate_information_content(self, ppm, eps = 1e-10):
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

    def calculate_pearson_correlation_coeffcient(self, ppm, S_one_hot):
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

    def calculate_alignment_score(self, ppm, S_one_hot):
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
        per_position_ic = self.calculate_information_content(ppm)
        per_position_pcc = self.calculate_pearson_correlation_coeffcient(ppm, S_one_hot)
        per_position_ic_weighted_pcc = per_position_pcc * (0.5 * per_position_ic)

        # Calculate the sum of the per_position_ic_weighted_pcc.
        ic_weighted_pcc_sum = np.sum(per_position_ic_weighted_pcc)

        return ic_weighted_pcc_sum

    def weighted_align(self, ppm, S_one_hot_na, S_non_x_mask):
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
                    if overlap_len < self.min_overlap_length or \
                       np.count_nonzero(S_non_x_mask_chunk) < self.min_overlap_length:
                        continue

                    # Remove any DX parts of the sequence and the corresponding
                    # part of the ppm.
                    ppm_chunk = ppm_chunk[S_non_x_mask_chunk]
                    S_one_hot_chunk = S_one_hot_chunk[S_non_x_mask_chunk]

                    score = self.calculate_alignment_score(ppm_chunk, S_one_hot_chunk)

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

    def align_ppms(self, ppms, S, chain_labels, protein_mask, dna_mask, rna_mask):
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
        """
        aligned_ppm = np.zeros((S.shape[0], len(self.restype_to_int)), dtype = np.float64)
        ppm_mask = np.zeros_like(S, dtype = np.int32)

        # Create a one-hot vector from the sequence.
        S_len = S.shape[0]
        S_one_hot = np.zeros((S_len, len(self.restype_to_int)), dtype = np.float64)
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
                na_restype_ints_to_compare = [self.restype_to_int["DA"], 
                                              self.restype_to_int["DC"], 
                                              self.restype_to_int["DG"],
                                              self.restype_to_int["DT"]]
            elif ppm_type == "rna":
                na_restype_ints_to_compare = [self.restype_to_int["A"], 
                                              self.restype_to_int["C"], 
                                              self.restype_to_int["G"],
                                              self.restype_to_int["U"]]
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
                 chain_opt_overlap_lens) = self.weighted_align(ppm, chain_S_one_hot_na, chain_S_non_x_mask)
                
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
                                ppm_score = self.calculate_alignment_score(ppm[ppm_idx][None, :], S_one_hot_na[S_idx][None, :])
                                aligned_ppm_score = self.calculate_alignment_score(aligned_ppm[S_idx, na_restype_ints_to_compare][None, :], S_one_hot_na[S_idx][None, :])
                                if ppm_score > aligned_ppm_score:
                                    aligned_ppm[S_idx, na_restype_ints_to_compare] = ppm[ppm_idx]
                            else:
                                ppm_col_ic = self.calculate_information_content(ppm[ppm_idx][None, :])
                                aligned_ppm_col_ic = self.calculate_information_content(aligned_ppm[S_idx, na_restype_ints_to_compare][None, :])
                                if ppm_col_ic > aligned_ppm_col_ic:
                                    aligned_ppm[S_idx, na_restype_ints_to_compare] = ppm[ppm_idx]
        
        return aligned_ppm, ppm_mask

    def load_chains(self, chains):
        #------------------proteins vs not--------------------
        macromolecule_letter_list = []
        for chain_letter, chain in chains.items():
            if chain.type == "polypeptide(L)":
                macromolecule_letter_list.append(chain_letter)
            elif chain.type == "polydeoxyribonucleotide":
                macromolecule_letter_list.append(chain_letter)
            elif chain.type == "polyribonucleotide":
                macromolecule_letter_list.append(chain_letter)
            elif chain.type == "polydeoxyribonucleotide/polyribonucleotide hybrid":
                macromolecule_letter_list.append(chain_letter)
  
        macromolecule_chain_dict = {}

        for letter in macromolecule_letter_list:
            chain = chains[letter]

            macromolecule_chain_dict[letter] = {}

            macromolecule_chain_dict[letter]["type"] = chain.type
        
            L = len(list(set([a[1] for a in list(chain.atoms.keys())])))
            xyz = np.zeros([L, self.num_atoms_to_save, 3], dtype=np.float32)
            occ = np.zeros([L, self.num_atoms_to_save], dtype=np.float32)
            residue_idx = -100*np.ones([L], dtype=np.int32) 
            raw_sequence = L*["UNK"]
            for c, (res_id, res_atoms) in enumerate(itertools.groupby(list(chain.atoms.keys()), lambda x: x[1])):
                for atom_key in res_atoms:
                    _, res_idx_str, res_name, atom_name = atom_key
                    
                    if atom_name in self.atom_dict:
                        atom_idx = self.atom_dict[atom_name]
                        xyz[c,atom_idx,:] = np.array(chain.atoms[atom_key].xyz)
                        occ[c,atom_idx] = np.array(chain.atoms[atom_key].occ)
                        # Same for each atom in the residue.
                        raw_sequence[c] = res_name
                        residue_idx[c] = int(res_idx_str)
            
            macromolecule_chain_dict[letter]["xyz"] = xyz
            macromolecule_chain_dict[letter]["occ"] = occ
            macromolecule_chain_dict[letter]["seq"] = raw_sequence
            macromolecule_chain_dict[letter]["residue_idx"] = residue_idx
        
        return macromolecule_chain_dict

    def load_assembly(self, macromolecule_chain_dict, asmb, assembly_id, ppms):
        X_list = []
        protein_mask_list = []
        dna_mask_list = []
        rna_mask_list = []
        X_occ_list = []
        S_list = []
        R_idx_list = []
        chain_labels_list = []
        chain_multi = 0
        
        for i, (letter, transform_matrix) in enumerate(asmb[assembly_id]):
            if letter in macromolecule_chain_dict:
                xyz = macromolecule_chain_dict[letter]["xyz"]
                rotation_matrix = transform_matrix[:3,:3]
                translation = transform_matrix[:3,3]
                xyz = np.einsum('ij,raj->rai', rotation_matrix, xyz) + translation[None,None,:]
                X_list.append(xyz)

                X_occ_list.append(macromolecule_chain_dict[letter]["occ"])

                R_idx_list.append(macromolecule_chain_dict[letter]["residue_idx"])

                chain_labels_list.append(chain_multi*np.ones_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32))
                chain_multi += 1

                protein_mask = np.zeros_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32)
                dna_mask = np.zeros_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32)
                rna_mask = np.zeros_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32)
                if macromolecule_chain_dict[letter]["type"] == "polypeptide(L)":
                    unknown_residue = "UNK"
                    protein_mask = np.ones_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32)
                elif macromolecule_chain_dict[letter]["type"] == "polydeoxyribonucleotide":
                    unknown_residue = "DX"
                    dna_mask = np.ones_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32) 
                elif macromolecule_chain_dict[letter]["type"] == "polyribonucleotide":
                    unknown_residue = "RX"
                    rna_mask = np.ones_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32)
                elif macromolecule_chain_dict[letter]["type"] == "polydeoxyribonucleotide/polyribonucleotide hybrid":
                    # Note, unknown residues in DNA/RNA hybrid chains are
                    # excluded from the DNA and RNA masks (it is not possible
                    # to know if a residue is one or the other due to the
                    # possibility of missing atoms). As such, the choice of
                    # unknown residue for the sequence does not matter.
                    unknown_residue = "DX"
                    for i, AA in enumerate(macromolecule_chain_dict[letter]["seq"]):
                        if AA in self.dna_restypes:
                            dna_mask[i] = 1
                        elif AA in self.rna_restypes:
                            rna_mask[i] = 1
                
                protein_mask_list.append(protein_mask)
                dna_mask_list.append(dna_mask)
                rna_mask_list.append(rna_mask)
                    
                seq_int = [self.restype_to_int.get(AA, self.restype_to_int[unknown_residue]) for AA in macromolecule_chain_dict[letter]["seq"]]
                S_list.append(np.array(seq_int, dtype=np.int32))

        X = np.concatenate(X_list, axis = 0) #[L, num_atoms, 3]
        X_occ = np.concatenate(X_occ_list, axis = 0) #[L, num_atoms]
        R_idx = np.concatenate(R_idx_list, axis = 0) #[L]
        chain_labels = np.concatenate(chain_labels_list, axis = 0) #[L]
        protein_mask = np.concatenate(protein_mask_list, axis = 0) #[L]
        dna_mask = np.concatenate(dna_mask_list, axis = 0) #[L]
        rna_mask = np.concatenate(rna_mask_list, axis = 0) #[L]
        S = np.concatenate(S_list, axis = 0) #[L]

        # Align ppm to the pre-cropped sequence.
        aligned_ppm, ppm_mask = self.align_ppms(ppms, S, chain_labels, 
                                                protein_mask, dna_mask, 
                                                rna_mask)

        R_polymer_type = protein_mask * self.polytype_to_int["PP"] + \
                    dna_mask * self.polytype_to_int["DNA"] + \
                    rna_mask * self.polytype_to_int["RNA"] + \
                    (1 - protein_mask - dna_mask - rna_mask) * self.polytype_to_int["UNK"]

        side_chain_occ_cutoff = protein_mask * self.protein_side_chain_occ_cutoff + \
                                dna_mask * self.dna_side_chain_occ_cutoff + \
                                rna_mask * self.rna_side_chain_occ_cutoff

        X_m = (X_occ > side_chain_occ_cutoff[:, None]).astype(np.int32)

        backbone_occ_cutoff = protein_mask * self.protein_backbone_occ_cutoff + \
                              dna_mask * self.dna_backbone_occ_cutoff + \
                              rna_mask * self.rna_backbone_occ_cutoff
        
        # Protein, DNA, and RNA masks are updated to only include residues with
        # all backbone atoms.
        X_occ_mask = (X_occ > backbone_occ_cutoff[:, None]).astype(np.int32)
        protein_mask = protein_mask * (np.prod(X_occ_mask[:, self.protein_bb_idx_list], axis = -1))
        dna_mask = dna_mask * (np.prod(X_occ_mask[:, self.dna_bb_idx_list], axis = -1))
        rna_mask = rna_mask * (np.prod(X_occ_mask[:, self.rna_bb_idx_list], axis = -1))

        if self.parse_rna_as_dna:
            dna_mask = np.bitwise_or(dna_mask, rna_mask)
            rna_mask = np.zeros_like(dna_mask)

        mask_for_output = np.zeros_like(protein_mask)
        out_dict = {}

        if self.parse_protein:
            mask_for_output = np.bitwise_or(mask_for_output, protein_mask)
            out_dict["protein_L"] = np.count_nonzero(protein_mask)
        else:
            out_dict["protein_L"] = 0
        
        if self.parse_dna:
            mask_for_output = np.bitwise_or(mask_for_output, dna_mask)
            out_dict["dna_L"] = np.count_nonzero(dna_mask)
        else:
            out_dict["dna_L"] = 0
        
        if self.parse_rna:
            mask_for_output = np.bitwise_or(mask_for_output, rna_mask)
            out_dict["rna_L"] = np.count_nonzero(rna_mask)
        else:
            out_dict["rna_L"] = 0
        
        out_dict["macromolecule_L"] = np.count_nonzero(mask_for_output)

        mask_for_output = mask_for_output.astype(bool)

        out_dict["protein_mask"] = protein_mask[mask_for_output]
        out_dict["dna_mask"] = dna_mask[mask_for_output]
        out_dict["rna_mask"] = rna_mask[mask_for_output]

        out_dict["X"] = X[mask_for_output]
        out_dict["X_m"] = X_m[mask_for_output]

        out_dict["S"] = S[mask_for_output]

        out_dict["R_idx"] = R_idx[mask_for_output]

        out_dict["chain_labels"] = chain_labels[mask_for_output]
        out_dict["R_polymer_type"] = R_polymer_type[mask_for_output]

        out_dict["aligned_ppm"] = aligned_ppm[mask_for_output]
        out_dict["ppm_mask"] = ppm_mask[mask_for_output]

        return out_dict

    def load_preprocessed_data(self, out_dict, example_dict, assembly_id):
        """
        Load any preprocessed data for the given example, specified by the
        example_dict and assembly id.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
                biomolecule.
            example_dict (dict): containing a dictionary that represents the
                column to value mapping of an example (a row from a dataframe).
            assembly_id (str): the id that specifies the assembly; needed for
                indexing into the assembly dictionaries in example_dict.
        
        Side Effects:
            out_dict['interface_mask']: the precomputed protein-nucleic acid
                interface mask.
            out_dict['side_chain_interface_mask']: the precomputed protein side
                chain-nucleic acid side chain interface mask.
            out_dict['nearest_protein_side_chain_index']: the precomputed index
                of the nearest protein side chain for each nucleic acid residue.
            out_dict['base_pair_mask']: the precomputed base pairing mask for
                nucleic acid residues.
            out_dict['base_pair_index']: the precomputed index of the base
                pairing partner for residues that are marked in the
                base_pair_mask.
            out_dict['canonical_base_pair_mask']: the precomputed canonical base
                pairing mask for nucleic acid residues.
            out_dict['canonical_base_pair_index']: the precomputed index of the
                canonical base pairing partner for residues that are marked in
                the canonical_base_pair_mask.
        """
        out_dict["interface_mask"] = \
            np.load(example_dict["asmb_interface_masks_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int32)
        out_dict["side_chain_interface_mask"] = \
            np.load(example_dict["asmb_side_chain_interface_masks_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int32)
        out_dict["nearest_protein_side_chain_index"] = \
            np.load(example_dict["asmb_nearest_protein_side_chain_index_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int64)
        out_dict["base_pair_mask"] = \
            np.load(example_dict["asmb_base_pair_masks_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int32)
        out_dict["base_pair_index"] = \
            np.load(example_dict["asmb_base_pair_index_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int64)
        out_dict["canonical_base_pair_mask"] = \
            np.load(example_dict["asmb_canonical_base_pair_masks_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int32)
        out_dict["canonical_base_pair_index"] = \
            np.load(example_dict["asmb_canonical_base_pair_index_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int64)

    def apply_crop_mask(self, out_dict, mask_to_keep):
        """
        Given a dictionary containing the loaded data for a biomolecule, and
        a mask of which residues to keep, crop all of the arrays of loaded
        data. For features that represent array indices, the indices need to
        be adjusted for the removed residues, and the associated masks need
        to be updated if the indices point to removed residues.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
                biomolecule.
            mask_to_keep (bool np.ndarray): a mask indicating which residues
                to keep when cropping. True at positions to keep, and False
                otherwise.

        Side Effects:
            out_dict[k]: cropped to only include the residues indicated by
                mask_to_keep, if k is an np.ndarray. If k denotes one of the
                index features, adjust the index for the removed residues, and
                if the index points to a removed residue, update the associated
                mask. If k denotes a macromolecule length, recalculate.
        """
        # Crop the loaded data.
        for k in out_dict:
            if type(out_dict[k]) == np.ndarray:
                out_dict[k] = out_dict[k][mask_to_keep]

        # Update variables that represent indices and associated masks.
        mask_to_remove = np.logical_not(mask_to_keep)
        index_of_removed = np.where(mask_to_remove)[0]
        residues_removed_to_left = np.array([0] + list(np.add.accumulate(mask_to_remove.astype(np.int32))[:-1]), dtype = np.int64)

        index_and_mask_key_pairs = [
            ("base_pair_index", "base_pair_mask"),
            ("canonical_base_pair_index", "canonical_base_pair_mask"),
            ("nearest_protein_side_chain_index", "side_chain_interface_mask")
        ]
        for (index_key, mask_key) in index_and_mask_key_pairs:
            index_in_removed = np.isin(out_dict[index_key], index_of_removed)

            # If the index that is pointed to was removed, mark the
            # corresponding position in the mask as 0.
            out_dict[mask_key][index_in_removed] = 0

            # For the indices that remain, subtract the residues removed to the
            # left of the position indicated by the index.
            out_dict[index_key] = out_dict[index_key] - residues_removed_to_left[out_dict[index_key]]
            out_dict[index_key] = out_dict[index_key] * out_dict[mask_key]

        # Update length data.
        out_dict["protein_L"] = np.count_nonzero(out_dict["protein_mask"])
        out_dict["dna_L"] = np.count_nonzero(out_dict["dna_mask"])
        out_dict["rna_L"] = np.count_nonzero(out_dict["rna_mask"])
        out_dict["macromolecule_L"] = out_dict["protein_L"] + out_dict["dna_L"] + out_dict["rna_L"]

    def drop_protein(self, out_dict):
        """
        Given a dictionary containing the loaded data for a biomolecule,
        drop all protein residues with a certain probability, dictated by
        self.drop_protein_probability.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
            biomolecule.

        Side Effects:
            out_dict[k]: crop to remove any protein residues. Set the interface
                masks to zero.
        """
        if sample_bernoulli_rv(self.drop_protein_probability) == 1:
            # Crop out the protein.
            not_protein_mask = np.logical_not(out_dict["protein_mask"] == 1)
            self.apply_crop_mask(out_dict, not_protein_mask)
            
            # Zero out the interface masks.
            out_dict["interface_mask"] = np.zeros_like(out_dict["interface_mask"])
            out_dict["side_chain_interface_mask"] = np.zeros_like(out_dict["side_chain_interface_mask"])

    def random_crop_na(self, out_dict):
        """
        Given a dictionary containing the loaded information of a biomolecule,
        crop the structure spatially around a randomly selected nucleic acid
        residue to the number of tokens in a batch.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
                biomolecule.

        Side Effects:
            out_dict[k]: crop the to the randomly selected, batch-sized spatial
                crop.
        """
        X = out_dict["X"]
        dna_mask = out_dict["dna_mask"]
        rna_mask = out_dict["rna_mask"]
        CA_idx = self.atom_dict["CA"]
        na_ref_atom_idx = self.atom_dict[self.na_ref_atom]

        ref_atom_X = X[:,CA_idx,:] + X[:,na_ref_atom_idx,:]

        # Choose a random nucleic acid to crop around.
        na_mask = dna_mask + rna_mask
        na_res_index = np.random.choice(np.where(na_mask == 1)[0])
        
        # Compute distance to all other residues.
        distance_to_na_res = np.sqrt(np.sum((ref_atom_X - ref_atom_X[na_res_index,:]) ** 2, axis = -1))
        argsorted_distance = np.argsort(distance_to_na_res)
        idx_to_keep = argsorted_distance[:self.batch_tokens]

        # Crop all array data.
        mask_to_keep = np.zeros_like(out_dict["S"], dtype=np.bool_)
        mask_to_keep[idx_to_keep] = True
        self.apply_crop_mask(out_dict, mask_to_keep)

    def uniformize_ppm_at_masked_positions(self, out_dict, mask_to_uniformize):
        """
        Given a dictionary containing the loaded information of a biomolecule,
        at the positions specified by mask_to_unformize, write 1 in the ppm_mask 
        and overwrite the ppm to be uniform across DA, DC, DG, DT for DNA or
        A, C, G, U for RNA.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
                biomolecule.
            mask_to_uniformize (bool np.ndarray): a L length array, that is True
                at nucleic acid positions to be uniformized and False otherwise.
        
        Side Effects:
            out_dict['aligned_ppm']: set to uniform (0.25) across DA, DC, DG, DT 
                for DNA or A, C, G, U for RNA for the positions indicated by 
                mask_to_uniformize.
            out_dict['ppm_mask']: set to 1 for uniformized nucleic acid 
                resiudes.
        """
        # Check that we are only uniformizing the nucleic acid positions.
        na_mask = np.logical_or(out_dict["dna_mask"] == 1,
                                out_dict["rna_mask"] == 1)
        assert(np.all(na_mask[mask_to_uniformize]))

        # Duplicate the aligned ppm and ppm mask.
        aligned_ppm = out_dict["aligned_ppm"].copy()
        ppm_mask = out_dict["ppm_mask"].copy()

        # Zero out and uniformize the aligned ppm at the specified positions.
        aligned_ppm[mask_to_uniformize] = 0

        # Set the DNA positions to uniform.
        mask_to_uniformize_dna = np.logical_and(mask_to_uniformize, out_dict["dna_mask"] == 1)
        aligned_ppm[mask_to_uniformize_dna, self.restype_to_int["DA"]] = 0.25
        aligned_ppm[mask_to_uniformize_dna, self.restype_to_int["DC"]] = 0.25
        aligned_ppm[mask_to_uniformize_dna, self.restype_to_int["DG"]] = 0.25
        aligned_ppm[mask_to_uniformize_dna, self.restype_to_int["DT"]] = 0.25

        # Set the RNA positions to uniform.
        mask_to_uniformize_rna = np.logical_and(mask_to_uniformize, out_dict["rna_mask"] == 1)
        aligned_ppm[mask_to_uniformize_rna, self.restype_to_int["A"]] = 0.25
        aligned_ppm[mask_to_uniformize_rna, self.restype_to_int["C"]] = 0.25
        aligned_ppm[mask_to_uniformize_rna, self.restype_to_int["G"]] = 0.25
        aligned_ppm[mask_to_uniformize_rna, self.restype_to_int["U"]] = 0.25

        # Set these positions as ppm positions.
        ppm_mask[mask_to_uniformize] = 1

        # Save the new aligned ppm and ppm mask.
        out_dict["aligned_ppm"] = aligned_ppm
        out_dict["ppm_mask"] = ppm_mask

    def uniformize_ppm_all_nucleic_acid(self, out_dict):
        """
        Given a dictionary containing the loaded information of a biomolecule,
        for all nucleic acid residues, write 1 in the ppm_mask and overwrite
        the ppm to be uniform across DA, DC, DG, DT for DNA or A, C, G, U for 
        RNA.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
                biomolecule.
        
        Side Effects:
            out_dict['aligned_ppm']: set to uniform (0.25) across DA, DC, DG, DT 
                for DNA or A, C, G, U for RNA for any nucleic acid residues.
            out_dict['ppm_mask']: set to 1 for all nucleic acid residues.
        """
        na_mask = np.logical_or(out_dict["dna_mask"] == 1,
                                out_dict["rna_mask"] == 1)
        
        self.uniformize_ppm_at_masked_positions(out_dict, na_mask)

    def uniformize_ppm_at_non_side_chain_interface(self, out_dict):
        """
        Given a dictionary containing the loaded information of a biomolecule,
        for all nucleic acid residues that do not have ppm information and are
        not located at the side chain interface, change the ppm to be uniform.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
                biomolecule.
        
        Side Effects:
            out_dict['aligned_ppm']: set to uniform (0.25) across across 
                DA, DC, DG, DT for DNA or A, C, G, U for RNA for any nucleic 
                acid residues without ppm information and not at the side chain 
                interface.
            out_dict['ppm_mask']: set to 1 for these residues.

        """
        na_mask = np.logical_or(out_dict["dna_mask"] == 1,
                                out_dict["rna_mask"] == 1)
        not_ppm_mask = np.logical_not(out_dict["ppm_mask"] == 1)
        not_side_chain_interface_mask = np.logical_not(out_dict["side_chain_interface_mask"] == 1)

        mask_to_uniformize = np.logical_and.reduce((na_mask, not_ppm_mask, not_side_chain_interface_mask))
        
        self.uniformize_ppm_at_masked_positions(out_dict, mask_to_uniformize)

    def mutate_interface_at_masked_positions(self, out_dict, mask_to_mutate):
        """
        Given a dictionary containing the loaded information of a biomolecule,
        and a mask indicating which protein side chain interface residues to 
        mutate, mutate the selected protein residues to a different sequence 
        identity and uniformize the ppm of any nucleic acid residues that were 
        closest to the mutated protein residues.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
                biomolecule.
            mask_to_mutate (bool np.ndarray): a L length array, that is True
                at protein side chain interface positions to be mutated and
                False otherwise.

        Side Effects:
            out_dict["S"]: randomly mutate to a different protein residue for
                the protein side chain interface residues specified in the 
                mask_to_mutate.
            out_dict['aligned_ppm']: set to uniform (0.25) across the
                appropriate residue types for the nucleic acid side chain 
                interface residues whose nearest protein side chain residue was 
                mutated.
            out_dict['ppm_mask']: set to 1 for the nucleic acid residues whose
                ppm was uniformized.
        """
        # Check that the residues to mutate are all protein side chain interface
        # residues.
        protein_side_chain_interface_mask = \
            np.logical_and(out_dict["protein_mask"] == 1, 
                           out_dict["side_chain_interface_mask"] == 1)
        assert(np.all(protein_side_chain_interface_mask[mask_to_mutate]))

        # Compute the nucleic acid side chain interface mask, for use in
        # selected the contacting nucleic acid residues.
        na_mask = np.logical_or(out_dict["dna_mask"] == 1,
                                out_dict["rna_mask"] == 1)
        na_side_chain_interface_mask = \
            np.logical_and(na_mask, 
                           out_dict["side_chain_interface_mask"] == 1)
    
        # For each protein residue to be mutated, mutate the residue and
        # uniformize contacting nucleic acid residue ppms.
        for protein_res_i in np.where(mask_to_mutate)[0]:
            contacting_na_residues = \
                list(np.where(np.logical_and(na_side_chain_interface_mask,
                                             out_dict["nearest_protein_side_chain_index"] == protein_res_i))[0])
            
            if self.mutate_base_pair_together:
                contacting_na_base_pair_residues = []
                for na_res_j in contacting_na_residues:
                    if out_dict["base_pair_mask"][na_res_j] == 1:
                        contacting_na_base_pair_residues.append(out_dict["base_pair_index"][na_res_j])
                contacting_na_residues = list(set(contacting_na_residues + contacting_na_base_pair_residues))

            if len(contacting_na_residues) > 0:
                # Mutate the protein residue.
                out_dict["S"][protein_res_i] = \
                    np.random.choice([res_int for res_int in self.protein_restype_ints if (res_int != out_dict["S"][protein_res_i] and res_int != self.restype_to_int["UNK"])])

                # Uniformize the PPM.
                for na_res_j in contacting_na_residues:
                    if out_dict["dna_mask"][na_res_j] == 1:
                        out_dict["aligned_ppm"][na_res_j, 
                                                [self.restype_to_int["DA"], 
                                                self.restype_to_int["DC"], 
                                                self.restype_to_int["DG"], 
                                                self.restype_to_int["DT"]]] = 0.25
                    elif out_dict["rna_mask"][na_res_j] == 1:
                        out_dict["aligned_ppm"][na_res_j, 
                                                [self.restype_to_int["A"], 
                                                self.restype_to_int["C"], 
                                                self.restype_to_int["G"], 
                                                self.restype_to_int["U"]]] = 0.25

                    out_dict["ppm_mask"][na_res_j] = 1

    def mutate_entire_side_chain_interface(self, out_dict):
        """
        Given a dictionary containing the loaded information of a biomolecule,
        with a probability dictated by 
        self.mutate_entire_side_chain_interface_probability, mutate all protein 
        side chain residues to a different sequence identity and uniformize the 
        ppm of all nucleic acid residues.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
                biomolecule.

        Side Effects:
            out_dict["S"]: randomly mutate to a different protein residue for
                all protein side chain interface residues.
            out_dict['aligned_ppm']: set to uniform (0.25) across the
                appropriate residue types for all nucleic acid residues.
            out_dict['ppm_mask']: set to 1 for the nucleic acid residues whose
                ppm was uniformized.
        """
        if sample_bernoulli_rv(self.mutate_entire_side_chain_interface_probability) == 1:
            protein_side_chain_interface_mask = \
                np.logical_and(out_dict["protein_mask"] == 1, 
                               out_dict["side_chain_interface_mask"] == 1)

            # Mutate the protein residues.
            self.mutate_interface_at_masked_positions(out_dict, protein_side_chain_interface_mask)

            # Ensure all nucleic acids are uniformized.
            self.uniformize_ppm_all_nucleic_acid(out_dict)

    def mutate_random_side_chain_interface(self, out_dict):
        """
        Given a dictionary containing the loaded information of a biomolecule,
        mutate all protein side chain residues randomly, with a per-residue
        probability dictated by 
        self.protein_interface_residue_mutation_probability, to a different 
        sequence identity and uniformize the ppm of any nucleic acid residues 
        that were closest to the mutated protein residues.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
                biomolecule.

        Side Effects:
            out_dict["S"]: randomly mutate to a different protein residue for
                the randomly selected protein side chain interface residues.
            out_dict['aligned_ppm']: set to uniform (0.25) across the
                appropriate residue types for the nucleic acid side chain 
                interface residues whose nearest protein side chain residue was 
                mutated.
            out_dict['ppm_mask']: set to 1 for the nucleic acid residues whose
                ppm was uniformized.
        """
        protein_side_chain_interface_mask = \
            np.logical_and(out_dict["protein_mask"] == 1, 
                           out_dict["side_chain_interface_mask"] == 1)
        
        per_residue_bernoulli_rvs = sample_bernoulli_rvs(p = self.protein_interface_residue_mutation_probability,
                                                         n = out_dict["macromolecule_L"])
        per_residue_mutation_mask = (per_residue_bernoulli_rvs == 1)

        interface_protein_residue_mutation_mask = \
            np.logical_and(per_residue_mutation_mask, 
                           protein_side_chain_interface_mask)

        self.mutate_interface_at_masked_positions(out_dict, interface_protein_residue_mutation_mask)

    def loader(self, example_dict, assembly_id):
        try:
            chains, asmb, covalei, meta = self.parse_structure(example_dict["structure_path"])
        except:
            print('bad_structure: ', example_dict["structure_path"])
            return ("pass", "pass")
        
        try:
            if self.parse_ppms:
                ppms, ppm_paths_chosen = self.load_ppms(example_dict["ppm_paths"], 
                                                        randomize_experimental_ppms = True)
            else:
                ppms, ppm_paths_chosen = self.load_ppms("[]", 
                                                        randomize_experimental_ppms = True)
        except:
            print('bad_ppms: ', example_dict["structure_path"], example_dict["ppm_paths"])
            return ("pass", "pass")
        
        if assembly_id not in list(asmb.keys()):
            print('bad_assembly_id: ', example_dict["structure_path"], assembly_id)
            return ("pass", "pass")

        macromolecule_chain_dict = self.load_chains(chains)

        out_dict = self.load_assembly(macromolecule_chain_dict, asmb, assembly_id, ppms)

        self.load_preprocessed_data(out_dict, example_dict, assembly_id)

        # Drop the protein with some probability.
        if self.drop_protein_probability > 0 and out_dict["macromolecule_L"] > out_dict["protein_L"]:
            self.drop_protein(out_dict)

        # Uniformize the ppms of free nucleic acid.
        if self.na_only_as_uniform_ppm and out_dict["protein_L"] == 0:
            self.uniformize_ppm_all_nucleic_acid(out_dict)
        
        # Uniformize the ppms of non side chain interface positions.
        if self.na_non_interface_as_uniform_ppm:
            self.uniformize_ppm_at_non_side_chain_interface(out_dict)

        # Randomly mutate side chain interface protein residues and modify the
        # contacting ppm positions to be uniform.
        if self.protein_interface_residue_mutation_probability > 0 and out_dict["protein_L"] > 0:
            self.mutate_random_side_chain_interface(out_dict)

        # Mutate all side chain interface protein residues and modify the
        # contacting ppm positions to be uniform.
        if self.mutate_entire_side_chain_interface_probability > 0 and out_dict["protein_L"] > 0:
            self.mutate_entire_side_chain_interface(out_dict)

        # Crop structures that are larger than the number of tokens in a batch.
        if self.crop_large_structures and out_dict["macromolecule_L"] > self.batch_tokens:
            self.random_crop_na(out_dict)
        
        out_dict["structure_path"] = example_dict["structure_path"]
        out_dict["assembly_id"] = assembly_id
        out_dict["ppm_paths"] = example_dict["ppm_paths"]
        out_dict["ppm_paths_chosen"] = ppm_paths_chosen

        return (out_dict, out_dict["macromolecule_L"])
    
    def load_for_structure_preprocessing(self, example_dict):
        try:
            chains, asmb, covalei, meta = self.parse_structure(example_dict["structure_path"])
        except:
            print('bad_structure: ', example_dict["structure_path"])
            return ("pass", "pass")

        # PPMs are not necessary for structure preprocessing.
        ppms = []

        # Save the per-chain sequences, for clustering purposes.
        chain_sequences = []
        for chain_letter in chains:
            chain = chains[chain_letter]
            chain_sequences.append((chain.id, chain.type, chain.sequence))

        macromolecule_chain_dict = self.load_chains(chains)

        assemblies = []
        for assembly_id in list(asmb.keys()):
            out_dict = self.load_assembly(macromolecule_chain_dict, asmb, assembly_id, ppms)
            assemblies.append((assembly_id, out_dict))
        
        return assemblies, chain_sequences

class StructureLoader():
    def __init__(self, dataset, macromolecule_lengths, max_tokens_per_batch):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = macromolecule_lengths
        self.max_tokens_per_batch = max_tokens_per_batch
        sorted_ix = np.argsort(self.lengths)
        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size > self.max_tokens_per_batch:
                continue

            if size * (len(batch) + 1) <= self.max_tokens_per_batch:
                batch.append(ix)
            else:
                if len(batch) > 0:
                    clusters.append(batch)
                batch = [ix]
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def make_batch_iter(df, batch_tokens, length_cutoff, date_cutoff, crop_large_structures, max_number_of_pdbs):
    """
    Creates an iterable batch for training or validation.

    Arguments:
        df (pd.DataFrame): a pandas DataFrame containing all information
            needed to load an example.
        batch_tokens (int): the maximum number of tokens (residues) in a batch.
        length_cutoff (int): the minimum macromolecule length for examples.
        date_cutoff (pd.DateTime): the date cutoff used for sampling; all
            samples past the date cutoff will be excluded.
        crop_large_structures (bool): if True, this indicates that large
            structures will be cropped to the batch size. If False, large
            structures (with more residues than the number of tokens in a batch)
            will be excluded.
        max_number_of_pdbs (int): the maximum number of PDBs to include in a
            batch.
    
    Returns:
        batch_iter (list_iterator): an iterable containing the row dictionaries
            for the samples in the batch.
    """
    samples=[]
    random_permutation = list(np.random.permutation(len(df)))
    for i in random_permutation:
        example_dict = df.iloc[i].to_dict()
        
        cluster_probability = example_dict["sampling_probability"]

        if (sample_bernoulli_rv(cluster_probability) == 1) and \
           (example_dict["date"] < date_cutoff): 
            samples.append(example_dict)

    L_list = []
    name_list = []
    for example_dict in samples:
        # Assembly ID to length dictionary.
        asmb_lengths_dict = np.load(example_dict["asmb_lengths_path"], allow_pickle = True).item()
        assembly_id_list = list(asmb_lengths_dict.keys())

        num_assemblies = len(assembly_id_list)
        if num_assemblies > 1:
            idx = np.random.randint(0, high = num_assemblies, dtype = int)
        else:
            idx = 0
        assembly_id = assembly_id_list[idx]

        (macromolecule_L, protein_L, dna_L, rna_L) = asmb_lengths_dict[assembly_id]

        if macromolecule_L >= length_cutoff and len(L_list) < max_number_of_pdbs:
            if macromolecule_L > batch_tokens and crop_large_structures and (dna_L + rna_L) > 0:
                macromolecule_L = batch_tokens
            L_list.append(macromolecule_L)
            name_list.append((example_dict, assembly_id))

    structure_loader = StructureLoader(name_list, L_list, max_tokens_per_batch=batch_tokens)

    batch_iter = []
    for _, batch in enumerate(structure_loader):
        batch_iter.append(batch)
    batch_iter = iter(batch_iter)
    return batch_iter
