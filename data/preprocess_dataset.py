import os
import numpy as np
import pandas as pd
import torch
import json

import sys
sys.path.append("/home/akubaney/projects/na_mpnn")

import pdbutils
import cifutils
from na_data_utils import PDBDataset

# Load the parameters file.
params = json.load(open("/home/akubaney/projects/na_mpnn/data/preprocess_dataset.json"))

# Load constants from the parameters file.
na_side_chain_atoms_len = len(['N9', 'C8', 'C7', 'N7', 'C6', 'N6', 'O6', 'C5', 'C4', 'N4', 'O4', 'N3', 'C2', 'N2', 'O2', 'N1'])
residue_cutoff = params["BATCH_TOKENS"]
num_neighbors = params["NUM_NEIGHBORS"]
interface_distance_cutoff = 5.0 # distance for interface in angstroms

if params["ATOMS_TO_LOAD"] == "backbone":
    atom_list_to_save = ['N', 'CA', 'C', 'O', #protein atoms
                         'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'" #nucleic acid atoms
                        ]
elif params["ATOMS_TO_LOAD"] == "all":
    atom_list_to_save = ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT', #protein atoms
                         'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N9', 'C8', 'C7', 'N7', 'C6', 'N6', 'O6', 'C5', 'C4', 'N4', 'O4', 'N3', 'C2', 'N2', 'O2', 'N1' #nucleic acid atoms
                        ]

# Create the parsers and dataset.
cif_parser = cifutils.CIFParser(skip_res=params["EXCLUDE_RES"], 
                                randomize_nmr_model=params["RANDOMIZE_NMR_MODEL"])
pdb_parser = pdbutils.PDBParser()

pdb_dataset = PDBDataset(cif_parser=cif_parser,
                         pdb_parser=pdb_parser,
                         atom_list_to_save=atom_list_to_save,
                         parse_protein=params["PARSE_PROTEIN"],
                         parse_dna=params["PARSE_DNA"],
                         parse_rna=params["PARSE_RNA"],
                         parse_rna_as_dna=params["PARSE_RNA_AS_DNA"],
                         na_shared_tokens=params["NA_SHARED_TOKENS"],
                         protein_backbone_occ_cutoff=params["PROTEIN_BACKBONE_OCC_CUTOFF"],
                         protein_side_chain_occ_cutoff=params["PROTEIN_SIDE_CHAIN_OCC_CUTOFF"],
                         dna_backbone_occ_cutoff=params["DNA_BACKBONE_OCC_CUTOFF"],
                         dna_side_chain_occ_cutoff=params["DNA_SIDE_CHAIN_OCC_CUTOFF"],
                         rna_backbone_occ_cutoff=params["RNA_BACKBONE_OCC_CUTOFF"],
                         rna_side_chain_occ_cutoff=params["RNA_SIDE_CHAIN_OCC_CUTOFF"],
                         crop_large_structures=params["CROP_LARGE_STRUCTURES"],
                         batch_tokens=params["BATCH_TOKENS"],
                         na_ref_atom=params["NA_REF_ATOM"]
                        )

# Create a mask for the side chain atoms.
side_chain_mask = np.zeros(len(pdb_dataset.atom_dict), dtype = np.int32) # [N]
for atom_name in pdb_dataset.atom_dict:
    if (atom_name not in pdb_dataset.protein_backbone_list) and \
       (atom_name not in pdb_dataset.dna_backbone_list) and \
       (atom_name not in pdb_dataset.rna_backbone_list):
        side_chain_mask[pdb_dataset.atom_dict[atom_name]] = 1

side_chain_pairwise_mask = side_chain_mask[:, None] * side_chain_mask[None, :] # [N, N]

def write_text_file(path, contents):
    with open(path, mode = "wt") as f:
        f.write(contents)

class HB_data:
    # Class modified from Andrew Favor.

    # amino acid type to integer
    num2aa=[
        'ALA','ARG','ASN','ASP','CYS',
        'GLN','GLU','GLY','HIS','ILE',
        'LEU','LYS','MET','PHE','PRO',
        'SER','THR','TRP','TYR','VAL',
        'UNK','MAS',
        ' DA',' DC',' DG',' DT', ' DX',
        ' RA',' RC',' RG',' RU', ' RX',
        'HIS_D', # only used for cart_bonded
        'Al', 'As', 'Au', 'B',
        'Be', 'Br', 'C', 'Ca', 'Cl',
        'Co', 'Cr', 'Cu', 'F', 'Fe',
        'Hg', 'I', 'Ir', 'K', 'Li', 'Mg',
        'Mn', 'Mo', 'N', 'Ni', 'O',
        'Os', 'P', 'Pb', 'Pd', 'Pr',
        'Pt', 'Re', 'Rh', 'Ru', 'S',
        'Sb', 'Se', 'Si', 'Sn', 'Tb',
        'Te', 'U', 'W', 'V', 'Y', 'Zn',
        'ATM'
    ]
    aa2num= {x:i for i,x in enumerate(num2aa)}
    aa2num['MEN'] = 20
    aa2num_stripped = {x.strip():i for i,x in enumerate(num2aa)}
    aa2num_stripped['MEN'] = 20
    
    # full sc atom representation
    NTOTAL = 36
    aa2long=[
        (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #0  ala
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), #1  arg
        (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), #2  asn
        (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), #3  asp
        (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), #4  cys
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), #5  gln
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), #6  glu
        (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), #7  gly
        (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","1HE ","2HE ",  None,  None,  None,  None,  None,  None), #8  his
        (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), #9  ile
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), #10 leu
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), #11 lys
        (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), #12 met
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","2HD ","1HE ","2HE "," HZ ",  None,  None,  None,  None), #13 phe
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), #14 pro
        (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), #15 ser
        (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), #16 thr
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE "," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), #17 trp
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE ","2HE ","2HD "," HH ",  None,  None,  None,  None), #18 tyr
        (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), #19 val
        (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #20 unk
        (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #21 mask

        (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," N9 "," C4 "," N3 "," C2 "," N1 "," C6 "," C5 "," N7 "," C8 "," N6 ",  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #22  DA
        (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," N1 "," C2 "," O2 "," N3 "," C4 "," N4 "," C5 "," C6 ",  None,  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #23  DC
        (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," N9 "," C4 "," N3 "," C2 "," N1 "," C6 "," C5 "," N7 "," C8 "," N2 "," O6 ",  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #24  DG
        (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," N1 "," C2 "," O2 "," N3 "," C4 "," O4 "," C5 "," C7 "," C6 ",  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H3 "," H71"," H72"," H73"," H6 ",  None), #25  DT
        (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'",  None,  None,  None,  None,  None,  None), #26  DX (unk DNA)
        (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'"," N1 "," C2 "," N3 "," C4 "," C5 "," C6 "," N6 "," N7 "," C8 "," N9 ",  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #27   A
        (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'"," N1 "," C2 "," O2 "," N3 "," C4 "," N4 "," C5 "," C6 ",  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #28   C
        (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'"," N1 "," C2 "," N2 "," N3 "," C4 "," C5 "," C6 "," O6 "," N7 "," C8 "," N9 "," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #29   G
        (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'"," N1 "," C2 "," O2 "," N3 "," C4 "," O4 "," C5 "," C6 ",  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H3 "," H5 "," H6 ",  None,  None,  None), #30   U
        (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'",  None,  None,  None,  None,  None,  None), #31  RX (unk RNA)

        (" N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","1HE ","1HD ",  None,  None,  None,  None,  None,  None), #-1 his_d
    ]
    aa2long_stripped = []
    for aa_tuple in aa2long:
        aa_tuple_stripped = tuple(map(lambda atom_name: atom_name.strip() if atom_name is not None else atom_name, aa_tuple))
        aa2long_stripped.append(aa_tuple_stripped)

    def __init__(self, seq, xyz, idx=None, **kwargs):
        # Required parameters
        self.seq = seq
        self.xyz = xyz
        
        if not idx:
            self.idx = torch.arange(len(seq))

        # Optional parameters with default values
        self.incl_protein = kwargs.get('incl_protein', True)
        self.eps = kwargs.get('eps', 1e-8)
        # self.use_eigennormals = kwargs.get('use_eigennormals', True)
        # self.use_all_base_atoms_for_MBD = kwargs.get('use_all_base_atoms_for_MBD', False)
        self.edges_to_compute = kwargs.get('edges_to_compute', ['S']) # list base edges to compute, if we want to analyze WC/Hoog/etc
        self.perp_base_edge = kwargs.get('perp_base_edge', 'S') # edge orthogonal to x- and z-directions in base frames (which is generally the sugar edge)

        self.hbond_da_upper = kwargs.get('hbond_da_upper', 3.9) 
        self.hbond_ha_upper = kwargs.get('hbond_ha_upper', 2.5)

        self.seq_cutoff = kwargs.get('seq_cutoff', 2)
        
        compute_local_base_params    = kwargs.get('compute_local_base_params', False)
        compute_pairwise_base_params = kwargs.get('compute_pairwise_base_params', False)
        compute_paired_bases         = kwargs.get('compute_paired_bases', False)
        compute_helical_params       = kwargs.get('compute_helical_params', False)


        
        self.base_geometry_limits = {}
        self.base_geometry_limits['D_ij'] = kwargs.get('D_ij_limit', 20.0)
        self.base_geometry_limits['H_ij'] = kwargs.get('H_ij_limit', 1.5)
        self.base_geometry_limits['P_ij'] = kwargs.get('P_ij_limit', np.pi/5)
        self.base_geometry_limits['B_ij'] = kwargs.get('B_ij_limit', np.pi/5)
        
        # self.base_geometry_limits['O_ij'] = kwargs.get('O_ij_limit', 1.5) # Not used right now, currently allow all values of opening
        self.bp_val_cutoff= kwargs.get('bp_val_cutoff', 0.5) # minimum basepairing score for using a pair when computing helical params
        
        
        
        self.hbond_same_coeff = kwargs.get('hbond_same_coeff', 0.0)
        self.hbond_diff_coeff = kwargs.get('hbond_diff_coeff', 1.0)
        self.min_hbonds_for_bp = kwargs.get('min_hbonds_for_bp', 2.0)
        self.bp_hbond_coeff  = kwargs.get('bp_hbond_coeff', 8.0)

        self.clamp_pairwise_params  = kwargs.get('clamp_pairwise_params', False)
        
        # Initialize computed attributes
        self._init_hb_chemdata()
        self._compute_initial_values()
        self._compute_hbnets(store_hb_data_dict=kwargs.get('store_hb_data_dict', False))
        
        # For now only doing for NA:
        if self.is_na.sum() > 0: # Only compute nucleic params when there are nucleics in the structure
            self._init_nuc_chemdata()
            self._edges_to_compute = list(set(self.edges_to_compute) | {self.perp_base_edge})  # Must compute this base-edge
            self._compute_local_base_params() # Define canonical base-frames for the specified edges
                
            if compute_pairwise_base_params or compute_paired_bases:
                self._compute_pairwise_base_params() # Compute pairwise geometric parameters between bases
                self._compute_paired_bases() # Classify bases using H-bond count and pairwise base geometry filters
            
            if compute_helical_params:
                self._compute_helical_params() # In progress...

    def _compute_initial_values(self):
        self.len_s = int(self.seq.shape[0])
        self.sel = torch.arange(self.len_s)
        self.seq_neighbors = torch.le(torch.abs(self.sel[:, None] - self.sel[None, :]), self.seq_cutoff)
        self.is_protein = torch.logical_and((0 <= self.seq), (self.seq <= 21))
        self.is_dna = torch.logical_and((22 <= self.seq), (self.seq <= 25))
        self.is_rna = torch.logical_and((27 <= self.seq), (self.seq <= 30))
        self.is_na = torch.logical_or(self.is_dna, self.is_rna)

        self.na_inds = [i for i,is_na_i in enumerate(self.is_na) if is_na_i]
        self.na_tensor_inds = {na_i:i for i,na_i in enumerate(self.na_inds)}
        
        frame_xyz = self.xyz[:,1,:]
        padded_centers = torch.cat([frame_xyz[:1], frame_xyz[:], frame_xyz[-1:]])
        
        self.D_ij_vec = frame_xyz.unsqueeze(0) - frame_xyz.unsqueeze(1) # pairwise displacement vector between frame centers
        self.D_ij = self.D_ij_vec.norm(dim=-1)
        self.M_i = ((padded_centers[1:-1] - padded_centers[:-2]) + (padded_centers[2:] - padded_centers[1:-1])) / 2 # average direction vector from consecutive frames in backbone
        self.M_i_doublet = padded_centers[1:] - padded_centers[:-1]
        
    def _compute_hbnets(self, store_hb_data_dict=False):
        
        # Distance between frames is between lower and upper bounds:
        D_ij_filter = (self.D_ij <= self.base_geometry_limits['D_ij'])
        
        # neighbor filter for all polymer types:
        neighbor_inds = torch.triu(D_ij_filter.bool(),diagonal=1).nonzero(as_tuple=True)
        
        pairwise_indices = list(zip(neighbor_inds[0].tolist(), neighbor_inds[1].tolist()))
        bp_pred_summation = torch.zeros_like(self.D_ij)

        # self.hb_data_dict = {i:{j:[] for j in range(self.len_s)} for i in range(self.len_s) }
        hb_data_dict = {i:{j:{} for j in range(self.len_s)} for i in range(self.len_s) }


        self.hbond_summation = torch.zeros_like(D_ij_filter, dtype=torch.float)

        for i,j in pairwise_indices:
            for a_i, is_donor_i in zip(self.hbond_atoms[HB_data.num2aa[self.seq[i]]]['names'],self.hbond_atoms[HB_data.num2aa[self.seq[i]]]['donor']):
                for a_j, is_donor_j in zip(self.hbond_atoms[HB_data.num2aa[self.seq[j]]]['names'],self.hbond_atoms[HB_data.num2aa[self.seq[j]]]['donor']):
                    atom_pair = f"{a_i}-{a_j}" # avoid duplicate counting for atom pairs
                    if (is_donor_i+is_donor_j)==1 and (atom_pair not in hb_data_dict[i][j].keys()):
                        

                        a_i_ind = HB_data.aa2long[self.seq[i]].index(a_i)
                        a_j_ind = HB_data.aa2long[self.seq[j]].index(a_j)

                        # Create vector between donor and acceptor atoms
                        d_ijk_vec = self.xyz[i,a_i_ind] - self.xyz[j,a_j_ind]
                        d_ijk_vec_norm = d_ijk_vec/d_ijk_vec.norm(dim=-1)

                        # Create vector giving direction to donor and acceptor along sidechain covalent bond:
                        a_i_vec = torch.cat(
                            [(self.xyz[i,a_i_ind]-self.xyz[i,HB_data.aa2long[self.seq[i]].index(r_i)])[:,None] for r_i in self.rear_atoms[HB_data.num2aa[self.seq[i]]][a_i]], 
                            dim=1).mean(dim=1)
                        a_i_vec_norm = a_i_vec/(a_i_vec.norm(dim=-1) + self.eps)

                        a_j_vec = torch.cat(
                            [(self.xyz[j,a_j_ind]-self.xyz[j,HB_data.aa2long[self.seq[j]].index(r_j)])[:,None] for r_j in self.rear_atoms[HB_data.num2aa[self.seq[j]]][a_j]], 
                            dim=1).mean(dim=1)
                        a_j_vec_norm = a_j_vec/(a_j_vec.norm(dim=-1) + self.eps)


                        num_rear_i = len(self.rear_atoms[HB_data.num2aa[self.seq[i]]][a_i])
                        element_i = ''.join([_ for _ in a_i if _.isalpha()])[0]
                        ideal_angle_i = self.ideal_angle_dict[element_i][num_rear_i]

                        num_rear_j = len(self.rear_atoms[HB_data.num2aa[self.seq[j]]][a_j])
                        element_j = ''.join([_ for _ in a_j if _.isalpha()])[0]
                        ideal_angle_j = self.ideal_angle_dict[element_j][num_rear_j]

                        ideal_angle_h = torch.tensor((is_donor_i*ideal_angle_i) + (is_donor_j*ideal_angle_j))

                        xyz_d_ijk = (  is_donor_i   * self.xyz[i,a_i_ind] ) + (  is_donor_j   * self.xyz[j,a_j_ind] )
                        xyz_a_ijk = ((1-is_donor_i) * self.xyz[i,a_i_ind] ) + ((1-is_donor_j) * self.xyz[j,a_j_ind] )

                        # (1, rd): vector pointing to donor atom from rear atom(s):
                        rd_ijk_vec = (is_donor_i * a_i_vec_norm) + (is_donor_j * a_j_vec_norm)
                        rd_ijk_vec_norm = rd_ijk_vec/(rd_ijk_vec.norm(dim=-1) + self.eps)

                        # (2, da): vector pointing from donor atom to acceptor atom, approximately in direction of the hydrogen:
                        da_ijk_vec = xyz_a_ijk - xyz_d_ijk
                        da_ijk_vec_norm = da_ijk_vec/(da_ijk_vec.norm(dim=-1) + self.eps)

                        # (3, ar): vector pointing to acceptor atom from rear atom(s):
                        ar_ijk_vec = ((is_donor_i-1)*a_i_vec_norm) +  ((is_donor_j-1)*a_j_vec_norm)
                        ar_ijk_vec_norm = ar_ijk_vec/(ar_ijk_vec.norm(dim=-1) + self.eps)


                        norm_vec = torch.cross(-rd_ijk_vec_norm, da_ijk_vec_norm, dim=-1)
                        norm_unit = norm_vec / (norm_vec.norm() + self.eps)  # Avoid divide-by-zero
                        perp_vec = torch.cross(norm_unit, -rd_ijk_vec_norm, dim=-1)
                        perp_unit = perp_vec / (perp_vec.norm() + self.eps)


                        # (4, dh): predicted ideal angle pointing from donor atom to hydrogen atom:
                        dh_ijk_vec = (torch.sin(ideal_angle_h) * perp_unit) - (torch.cos(ideal_angle_h) * rd_ijk_vec_norm)
                        dh_ijk_vec_norm = dh_ijk_vec / (dh_ijk_vec.norm() + self.eps) # norm actually matters here, because Donor -> H distance is exactly 1A.
                        ideal_xyz_h_ijk = xyz_d_ijk + dh_ijk_vec_norm # Compute ideal hydrogen placement

                        # (5, ha): vector pointing from ideal hydrogen to acceptor atom
                        ha_ijk_vec = xyz_a_ijk - ideal_xyz_h_ijk
                        ha_ijk_vec_norm = ha_ijk_vec / (ha_ijk_vec.norm() + self.eps)


                        t_rdh = torch.acos( ( -rd_ijk_vec_norm * dh_ijk_vec_norm ).sum(dim=-1) )
                        t_rda = torch.acos( ( -rd_ijk_vec_norm * da_ijk_vec_norm ).sum(dim=-1) )
                        t_dha = torch.acos( ( -dh_ijk_vec_norm * ha_ijk_vec_norm ).sum(dim=-1) )
                        t_dar = torch.acos( ( -da_ijk_vec_norm * ar_ijk_vec_norm ).sum(dim=-1) )
                        t_har = torch.acos( ( -ha_ijk_vec_norm * ar_ijk_vec_norm ).sum(dim=-1) )


                        da_ijk = da_ijk_vec.norm(dim=-1)
                        ha_ijk = ha_ijk_vec.norm(dim=-1)

                        hbond_da_filter = ( da_ijk <= self.hbond_da_upper )
                        hbond_ha_filter = ( ha_ijk <= self.hbond_ha_upper ) # SHOULD BE MOST IMPORTANT
                        
                        hbond_t_rda_filter = ( t_rda >= 5*np.pi/9 ) #   cutoff (100 degrees) proposed by:   https://pmc.ncbi.nlm.nih.gov/articles/PMC8261469/
                        hbond_t_dar_filter = ( t_dar >= 5*np.pi/9 ) #   similar logic to above
                        hbond_t_dha_filter = ( t_dha >= np.pi/2   ) #   could also increase this one maybe
                        
                        bond_prob_ij = (hbond_ha_filter * hbond_da_filter * hbond_t_rda_filter * hbond_t_dar_filter).float()

                        self.hbond_summation[i,j] += bond_prob_ij
                        self.hbond_summation[j,i] += bond_prob_ij
                        
                        hb_data_dict[i][j][atom_pair] = {'d': da_ijk, 'l': ha_ijk, "t_rdh": t_rdh, "t_rda": t_rda, "t_dha": t_dha, "t_dar": t_dar, "t_har": t_har, 'atoms': atom_pair, "bonded": bond_prob_ij, }
                        hb_data_dict[j][i][atom_pair] = {'d': da_ijk, 'l': ha_ijk, "t_rdh": t_rdh, "t_rda": t_rda, "t_dha": t_dha, "t_dar": t_dar, "t_har": t_har, 'atoms': atom_pair, "bonded": bond_prob_ij, }

        if store_hb_data_dict:
            self.hb_data_dict = hb_data_dict
                        
    def _compute_local_base_params(self):
        """
        local base params , based on interaction-edges

        """
        xyz_na = self.xyz[self.is_na]
        seq_na = self.seq[self.is_na]
        

        """
        (1). Compute base normals and correct orientation based on backbone direction.
        """
        base_atom_xyz = torch.stack([xyz_na[i,self.ring_atom_inds[HB_data.num2aa[s_i]],:] for i,s_i in enumerate(seq_na)] )
        base_atom_centers = torch.mean(base_atom_xyz, dim=1)
            
        centered_points = base_atom_xyz - base_atom_centers.unsqueeze(1)
        cov_matrix = torch.einsum('bij,bik->bjk', centered_points, centered_points) / (centered_points.shape[1] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Keep N_i local, since we will only need Z_i after this function
        N_i = eigenvectors[:, :, 0] / eigenvectors[:, :, 0].norm(dim=1, keepdim=True) 

        # Correct base normals to point in direction of backbone 5' -> 3' by projecting backbone vec M_i onto this unit Z_i vector to flip direction if necessary
        self.Z_i = N_i * torch.sum(self.M_i[self.is_na] * N_i, dim=-1, keepdim=True)
        self.Z_i = self.Z_i / (torch.norm(self.Z_i, dim=-1, keepdim=True) + self.eps)
        
        """
        (2). Compute the desired edge-vectors for the bases (watson-crick, hoogstein, sugar, etc)
            * W edge: N1 of purine, N3  of pyrimidine
            * H edge: N7 of purine, C5  of pyrimidine
            * S edge: N3 of purine, C1' of pyrimidine
            * B (pseudo)-edge: connects C1' to first base-atom (N1 or N3?)
        """
        # Compute X and Y vectors for edges of interest:
        self.edge_X_vecs, self.edge_Y_vecs = {}, {}
        for edge in self.edges_to_compute:
            self.edge_X_vecs[edge]     = torch.stack([xyz_na[i,self.vec_atom_inds[HB_data.num2aa[s_i]][f'{edge}_stop'],:] - xyz_na[i,self.vec_atom_inds[HB_data.num2aa[s_i]][f'{edge}_start'],:] for i, s_i in  enumerate(seq_na)])
            self.edge_X_vecs[edge]     = self.edge_X_vecs[edge] / (torch.norm(self.edge_X_vecs[edge], dim=-1, keepdim=True) + self.eps)
            
            # self.edge_Y_vecs[edge]     = torch.cross(self.edge_X_vecs[edge], N_i, dim=-1)
            self.edge_Y_vecs[edge]     = torch.cross(self.edge_X_vecs[edge], self.Z_i, dim=-1)
            self.edge_Y_vecs[edge]     = self.edge_Y_vecs[edge] / (torch.norm(self.edge_Y_vecs[edge], dim=-1, keepdim=True) + self.eps)
        
        """
        (3).  Define canonical base frames in terms of one specific edge.
              The sugar edge generally works best here, as it most often points towards binding partner 
              (orthogonal to both major groove and helical axis)
        """
        
        self.X_i = torch.cross(self.Z_i, self.edge_X_vecs[self.perp_base_edge], dim=-1)
        self.X_i = self.X_i / (torch.norm(self.X_i, dim=-1, keepdim=True) + self.eps)

        self.Y_i = torch.cross(self.X_i, self.Z_i, dim=-1)
        self.Y_i = self.Y_i / (torch.norm(self.Y_i, dim=-1, keepdim=True) + self.eps)
        self.base_atom_centers = base_atom_centers
        
    def _compute_pairwise_base_params(self):
        
        D_ij_vec_na = self.D_ij_vec[torch.arange(self.is_na.sum()).unsqueeze(1), torch.arange(self.is_na.sum())]
        base_D_ij_vec = self.base_atom_centers.unsqueeze(0) - self.base_atom_centers.unsqueeze(1)
        
        # stack mean Z-direction vectors for parallel (0) and antiparallel (1) orientations in zeroth-axis:
        Z_ij_oris = 0.5*torch.stack((self.Z_i.unsqueeze(1) + self.Z_i.unsqueeze(0), self.Z_i.unsqueeze(1) - self.Z_i.unsqueeze(0) ), dim=0)

        # Check which are parallel or antiparallel:
        bases_are_antiparallel = (Z_ij_oris[1].norm(dim=-1) > Z_ij_oris[0].norm(dim=-1)).long()

        # Extract mean Z-direction based on maximum shared direction between planes of base i and j:
        Z_ij = Z_ij_oris[bases_are_antiparallel, torch.arange(self.is_na.sum()).unsqueeze(1), torch.arange(self.is_na.sum())]
        Z_ij = Z_ij / (torch.norm(Z_ij, dim=-1, keepdim=True) + self.eps)
        
        Y_ij = D_ij_vec_na / (torch.norm(D_ij_vec_na, dim=-1, keepdim=True) + self.eps)
        X_ij = torch.cross(Z_ij, Y_ij, dim=-1)

        X_ij = X_ij / (torch.norm(X_ij, dim=-1, keepdim=True) + self.eps)
        
        self.H_ij = torch.sum(base_D_ij_vec * Z_ij, dim=-1)
        self.H_ij_vec = self.H_ij[...,None] * Z_ij

        # Opening: angle between local x_i and x_j within global X_ij-Y_ij plane:
        proj_X_i_XY = ((self.X_i[:, None, :] * X_ij).sum(dim=-1, keepdim=True) * X_ij) + ((self.X_i[:, None, :] * Y_ij).sum(dim=-1, keepdim=True) * Y_ij)
        proj_X_i_XY_norm = proj_X_i_XY / (torch.norm(proj_X_i_XY, dim=-1, keepdim=True) + self.eps)
        cos_opening = (proj_X_i_XY_norm * proj_X_i_XY_norm.transpose(1,0)).sum(dim=-1)
        if self.clamp_pairwise_params:
            cos_opening = torch.clamp(cos_opening, -1.0, 1.0)
        O_ij = torch.acos(cos_opening)

        # Buckle: angle between local z_i and z_j within global Y_ij-Z_ij plane:
        proj_Z_i_YZ = ((self.Z_i[:, None, :] * Y_ij).sum(dim=-1, keepdim=True) * Y_ij) + ((self.Z_i[:, None, :] * Z_ij).sum(dim=-1, keepdim=True) * Z_ij)    
        proj_Z_i_YZ_norm = proj_Z_i_YZ / (torch.norm(proj_Z_i_YZ, dim=-1, keepdim=True) + self.eps)
        cos_buckle = (proj_Z_i_YZ_norm * -proj_Z_i_YZ_norm.transpose(1,0)).sum(dim=-1)
        if self.clamp_pairwise_params:
            cos_buckle = torch.clamp(cos_buckle, -1.0, 1.0)
        cos_buckle = torch.clamp(cos_buckle, -1.0, 1.0)
        B_ij = torch.acos(cos_buckle)

        # Propeller: angle between local z_i and z_j within global Z_ij-X_ij plane:
        proj_Z_i_ZX = ((self.Z_i[:, None, :] * Z_ij).sum(dim=-1, keepdim=True) * Z_ij) + ((self.Z_i[:, None, :] * X_ij).sum(dim=-1, keepdim=True) * X_ij)    
        proj_Z_i_ZX_norm = proj_Z_i_ZX / (torch.norm(proj_Z_i_ZX, dim=-1, keepdim=True) + self.eps)
        cos_propeller = (proj_Z_i_ZX_norm * -proj_Z_i_ZX_norm.transpose(1,0)).sum(dim=-1)
        if self.clamp_pairwise_params:
            cos_propeller = torch.clamp(cos_propeller, -1.0, 1.0)
        P_ij = torch.acos(cos_propeller)
        
        # Local frame components for sidechains:
        self.X_ij = X_ij
        self.Y_ij = Y_ij
        self.Z_ij = Z_ij
        
        # pairwise base parameters:
        self.O_ij = O_ij
        self.B_ij = B_ij
        self.P_ij = P_ij
        self.bases_are_antiparallel = bases_are_antiparallel

    def _compute_paired_bases(self):
        
        # Compute baseline bp probability based on hydrogen bond count
        bp_preds = torch.sigmoid(self.bp_hbond_coeff * (self.hbond_summation - (self.min_hbonds_for_bp - 1))) # offset by 1 for midpoint
        
        # basepair-specific filters:
        both_nucleic_filter = self.is_na[:,None] * self.is_na[None,:]
        
        # Frame distance filter, already taken care of
        # D_ij_filter = (self.D_ij_vec.norm(dim=-1) < self.base_geometry_limits['D_ij']) 
        # Rise between base-planes is within lower and upper bounds
        # H_ij_filter = (self.H_ij.norm(dim=-1) > -self.base_geometry_limits['H_ij']) & (self.H_ij.norm(dim=-1) < self.base_geometry_limits['H_ij']) 
        H_ij_filter = (self.H_ij >= -self.base_geometry_limits['H_ij']) & (self.H_ij <= self.base_geometry_limits['H_ij']) 
        # H_ij_filter = (self.H_ij.T >= -self.base_geometry_limits['H_ij']) & (self.H_ij.T <= self.base_geometry_limits['H_ij']) 
        # Buckle between bases is either lower than lower bound or higher than upper bound (stay close to 0 or pi):
        B_ij_filter = (self.B_ij <= (np.pi - self.base_geometry_limits['B_ij'])) | (self.B_ij >= self.base_geometry_limits['B_ij'])
        # Propeller between bases is either lower than lower bound or higher than upper bound (stay close to 0 or pi):
        P_ij_filter = (self.P_ij <= (np.pi - self.base_geometry_limits['P_ij'])) | (self.P_ij >= self.base_geometry_limits['P_ij'])
        
        
        # combine for full basepair filter:
        bp_geom_filter = torch.zeros(self.len_s, self.len_s, dtype=torch.bool)
        bp_geom_filter[torch.outer(self.is_na, self.is_na)] = (H_ij_filter * B_ij_filter * P_ij_filter).flatten()
        # bp_geom_filter[torch.outer(self.is_na, self.is_na)] = (              B_ij_filter * P_ij_filter).flatten()
        # bp_geom_filter[torch.outer(self.is_na, self.is_na)] = (H_ij_filter *               P_ij_filter).flatten()
        # bp_geom_filter[torch.outer(self.is_na, self.is_na)] = (H_ij_filter * B_ij_filter              ).flatten()
        self.basepairs_ij = both_nucleic_filter * bp_geom_filter * bp_preds

    def _compute_helical_params(self):
        
        len_na = self.Z_i.shape[0] # Do I need this?
        nucleic_frames = self.xyz[self.is_na, 1, :]
        doublet_inds = [(i,j) for (i,j) in zip(range(0,len_na-1),range(1,len_na))]
        
        
        Zm_i = torch.zeros_like(self.Z_i)
        Zh_i = torch.zeros_like(self.Z_i)
        
        # Local doublet step params
        tilt_i          = torch.zeros(len_na)
        roll_i          = torch.zeros(len_na)
        twist_i         = torch.zeros(len_na)
        shift_i         = torch.zeros(len_na)
        slide_i         = torch.zeros(len_na)
        rise_i          = torch.zeros(len_na)
        
        
        # Local helical parameters
        inclination_i   = torch.zeros(len_na)
        tip_i           = torch.zeros(len_na)
        helical_twist_i = torch.zeros(len_na)
        x_disp_i        = torch.zeros(len_na)
        y_disp_i        = torch.zeros(len_na)
        helical_rise_i  = torch.zeros(len_na)
        
        # avg_factor = torch.zeros_like(self.Z_i[:,0])
        avg_factor = torch.zeros(len_na)
        for i,j in doublet_inds:
            avg_factor[i] += 1.0
            avg_factor[j] += 1.0
        
        
        basepaired_inds = (self.basepairs_ij >= self.bp_val_cutoff).bool().nonzero(as_tuple=True)
        
        pairwise_indices = list(zip(basepaired_inds[0].tolist(), basepaired_inds[1].tolist()))
        
        # partner_info_dict = {i:{'partner_ind':[], 'orientation':[], 'num_hbonds':[], 'bp_score': []} for i in range(len_na)}
        partner_info_dict = {i:{'partner_ind':[], 'orientation':[], 'num_hbonds':[], 'bp_score': []} for i in self.na_inds}
        
        for i, j in pairwise_indices:
            _i,_j = self.na_tensor_inds[i], self.na_tensor_inds[j]
            partner_info_dict[i]['partner_ind'].append(j)
            partner_info_dict[i]['orientation'].append(self.bases_are_antiparallel[_i,_j])
            partner_info_dict[i]['num_hbonds'].append(self.hbond_summation[_i,_j])
            partner_info_dict[i]['bp_score'].append(self.basepairs_ij[_i,_j])
            
                


        # TO DO: sort partner_info_dict[i]['partner_ind'] list by: first by orientation, then by num_hbonds
        # If we don't do the sorting, no need to compile these lists in the dict. Can just index directly from precomputed values.
        # for i in partner_info_dict.keys():
        for i_1, i_2 in doublet_inds:

            partners_i_1 = [self.na_tensor_inds[_] for _ in partner_info_dict[self.na_inds[i_1]]['partner_ind']]
            partners_i_2 = [self.na_tensor_inds[_] for _ in partner_info_dict[self.na_inds[i_2]]['partner_ind']]

                
            # set_trace()
            # j_1 = partners_i_1[0] # index-[0] is just a placeholder for later iteration
            # j_2 = partners_i_2[0] # index-[0] is just a placeholder for later iteration
            num_partners_i_1 = len(partners_i_1) # later change to be length of iterable
            num_partners_i_2 = len(partners_i_2) # later change to be length of iterable
            for j_1 in partners_i_1:
                # _j_1 = self.na_tensor_inds[j_1]
                for j_2 in partners_i_2:
                    X_1 = self.X_ij[i_1,j_1]
                    Y_1 = self.Y_ij[i_1,j_1]
                    X_2 = self.X_ij[i_2,j_2]
                    Y_2 = self.Y_ij[i_2,j_2]

                    Xp = X_2 + X_1
                    Xn = X_2 - X_1
                    Yp = Y_2 + Y_1
                    Yn = Y_2 - Y_1
                    
                    M_12 = 0.5*((nucleic_frames[i_2]+nucleic_frames[j_2]) - (nucleic_frames[i_1]+nucleic_frames[j_1]))

                    Zm = torch.cross(Xp, Yp, dim=-1) / ((Xp.norm(dim=-1) * Yp.norm(dim=-1)) + self.eps)
                    Zh = torch.cross(Xn, Yn, dim=-1) / ((Xn.norm(dim=-1) * Yn.norm(dim=-1)) + self.eps)

                    Zm_i[i_1] += Zm / (avg_factor[i_1]+self.eps)
                    Zh_i[i_1] += Zh / (avg_factor[i_1]+self.eps)

                    Zm_i[i_2] += Zm / (avg_factor[i_2]+self.eps)
                    Zh_i[i_2] += Zh / (avg_factor[i_2]+self.eps)


                    tilt_ij  = -torch.arcsin(torch.sum(Zm * X_1 , dim=-1))
                    roll_ij  =  torch.arcsin(torch.sum(Zm * Y_1 , dim=-1))
                    twist_ij = torch.arccos(torch.sum(torch.cross(X_1 , Zm, dim=-1) * torch.cross(X_2 , Zm, dim=-1), dim=-1))
                    shift_ij = torch.sum(M_12 * (Xp / (torch.norm(Xp, dim=-1)+self.eps)), dim=-1)
                    slide_ij = torch.sum(M_12 * (Yp / (torch.norm(Yp, dim=-1)+self.eps)), dim=-1)
                    rise_ij  = torch.sum(M_12 * Zm , dim=-1)

                    inclination_ij    =  torch.arcsin(torch.sum(Zh * X_1 , dim=-1))
                    tip_ij            = -torch.arcsin(torch.sum(Zh * Y_1 , dim=-1))
                    helical_twist_ij  = -torch.arccos(torch.sum(torch.cross(X_1 , Zh, dim=-1) * torch.cross(X_2 , Zh, dim=-1), dim=-1))
                    x_disp_ij         =  torch.sum(M_12 * Xn / (torch.norm(Xn, dim=-1)+self.eps), dim=-1)
                    y_disp_ij         =  torch.sum(M_12 * Yn / (torch.norm(Yn, dim=-1)+self.eps), dim=-1)
                    helical_rise_ij   = -torch.sum(M_12 * Zh, dim=-1)

                    # NEXT JUST ADD THESE PARAMS TO SOME PRE-INITIALIZED TENSOR AND DIVIDE BY AVG_FACTOR TO AVERAGE:
                    # For doublet position-1:
                    avg_factor[i_1] += 1.0
                    tilt_i[i_1] +=           tilt_ij          
                    roll_i[i_1] +=           roll_ij          
                    twist_i[i_1] +=          twist_ij         
                    shift_i[i_1] +=          shift_ij         
                    slide_i[i_1] +=          slide_ij         
                    rise_i[i_1] +=           rise_ij          
                    inclination_i[i_1] +=    inclination_ij   
                    tip_i[i_1] +=            tip_ij           
                    helical_twist_i[i_1] +=  helical_twist_ij 
                    x_disp_i[i_1] +=         x_disp_ij        
                    y_disp_i[i_1] +=         y_disp_ij        
                    helical_rise_i[i_1] +=   helical_rise_ij  

                    # For doublet position-2:
                    avg_factor[i_2] += 1.0
                    tilt_i[i_2] +=           tilt_ij          
                    roll_i[i_2] +=           roll_ij          
                    twist_i[i_2] +=          twist_ij         
                    shift_i[i_2] +=          shift_ij         
                    slide_i[i_2] +=          slide_ij         
                    rise_i[i_2] +=           rise_ij          
                    inclination_i[i_2] +=    inclination_ij   
                    tip_i[i_2] +=            tip_ij           
                    helical_twist_i[i_2] +=  helical_twist_ij 
                    x_disp_i[i_2] +=         x_disp_ij        
                    y_disp_i[i_2] +=         y_disp_ij        
                    helical_rise_i[i_2] +=   helical_rise_ij  


                self.tilt_i          = tilt_i / (avg_factor + self.eps)
                self.roll_i          = roll_i / (avg_factor + self.eps)
                self.twist_i         = twist_i / (avg_factor + self.eps)
                self.shift_i         = shift_i / (avg_factor + self.eps)
                self.slide_i         = slide_i / (avg_factor + self.eps)
                self.rise_i          = rise_i / (avg_factor + self.eps)
                self.inclination_i   = inclination_i / (avg_factor + self.eps)
                self.tip_i           = tip_i / (avg_factor + self.eps)
                self.helical_twist_i = helical_twist_i / (avg_factor + self.eps)
                self.x_disp_i        = x_disp_i / (avg_factor + self.eps)
                self.y_disp_i        = y_disp_i / (avg_factor + self.eps)
                self.helical_rise_i  = helical_rise_i / (avg_factor + self.eps)

    def _init_hb_chemdata(self):



        #     RESIDUE   |       DONORS        |     ACCEPTORS     
        self.hbond_atoms = {
            "ALA": {"names":[                                                        ],
                    "donor":[                                                        ]},
            "ARG": {"names":[" NH1"," NH2"                                           ],
                    "donor":[   1  ,   1                                             ]},
            "ASN": {"names":[" ND2",               " OD1"                            ],
                    "donor":[   1  ,                  0                              ]},
            "ASP": {"names":[" OD2",               " OD1"," OD2"                     ],
                    "donor":[   1  ,                  0  ,   0                       ]},
            "CYS": {"names":[" SG "                                                  ],
                    "donor":[   1                                                    ]},
            "GLN": {"names":[" NE2",               " OE1"                            ],
                    "donor":[   1  ,                  0                              ]},
            "GLU": {"names":[" OE2",               " OE1"," OE2"                     ],
                    "donor":[   1  ,                  0  ,   0                       ]},
            "GLY": {"names":[                                                        ],
                    "donor":[                                                        ]},
            "HIS": {"names":[" ND1"," NE2",        " ND1"," NE2"                     ],
                    "donor":[   1  ,   1  ,           0  ,   0                       ]},
            "ILE": {"names":[                                                        ],
                    "donor":[                                                        ]},
            "LEU": {"names":[                                                        ],
                    "donor":[                                                        ]},
            "LYS": {"names":[" NZ "                                                  ],
                    "donor":[   1                                                    ]},
            "MET": {"names":[                      " SD "                            ],
                    "donor":[                         0                              ]},
            "PHE": {"names":[                                                        ],
                    "donor":[                                                        ]},
            "PRO": {"names":[                                                        ],
                    "donor":[                                                        ]},
            "SER": {"names":[" OG "                                                  ],
                    "donor":[   1                                                    ]},
            "THR": {"names":[" OG1"                                                  ],
                    "donor":[   1                                                    ]},
            "TRP": {"names":[                      " NE1"                            ],
                    "donor":[                         0                              ]},
            "TYR": {"names":[" OH "                                                  ],
                    "donor":[   1                                                    ]},
            "VAL": {"names":[                                                        ],
                    "donor":[                                                        ]},
            "UNK": {"names":[                                                        ],
                    "donor":[                                                        ]},
            "MAS": {"names":[                                                        ],
                    "donor":[                                                        ]},
            " DA": {"names":[" N6 ",               " N1 "," N3 "," N7 "              ],
                    "donor":[   1  ,                  0  ,   0  ,   0                ]},
            " DG": {"names":[" N1 "," N2 "," N7 ", " O6 "," N1 "," N3 "," N7 "       ],
                    "donor":[   1  ,   1  ,   1  ,    0  ,   0  ,   0  ,   0         ]},
            " DC": {"names":[" N4 "," N3 ",        " O2 "," N3 "                     ],
                    "donor":[   1  ,   1  ,           0  ,   0                       ]},
            " DT": {"names":[" N3 ",               " O2 "," O4 "                     ],
                    "donor":[   1  ,                  0  ,   0                       ]},
            " DX": {"names":[                                                        ],
                    "donor":[                                                        ]},
            " RA": {"names":[" O2'"," N6 ",        " N1 "," N3 "," N7 "              ],
                    "donor":[    1 ,   1  ,           0  ,   0  ,   0                ]},
            " RG": {"names":[" O2'"," N1 "," N2 "," N7 ", " O6 "," N1 "," N3 "," N7 "],
                    "donor":[    1 ,   1  ,   1  ,   1  ,    0  ,   0  ,   0  ,   0  ]},
            " RC": {"names":[" O2'"," N4 "," N3 ",        " O2 "," N3 "              ],
                    "donor":[    1 ,   1  ,   1  ,           0  ,   0                ]},
            " RU": {"names":[" O2'"," N3 ",               " O2 "," O4 "              ],
                    "donor":[    1 ,    1 ,                  0  ,   0                ]},
            " RX": {"names":[" O2'",                                                 ],
                    "donor":[    1 ,                                                 ]},
        }



        # define atoms behind all donors/acceptors/tip-atoms so that we can use them to draw a vector giving the direction of [rear-atoms] -> [tip-atoms] 
        self.rear_atoms = {
            "ALA": {},
            "ARG": {" NH1":[" CZ "], " NH2":[" CZ "],},
            "ASN": {" OD1":[" CG "], " ND2":[" CG "],},
            "ASP": {" OD1":[" CG "], " OD2":[" CG "],},
            "CYS": {" SG ":[" CB "],},
            "GLN": {" OE1":[" CD "], " NE2":[" CD "],},
            "GLU": {" OE1":[" CD "], " OE2":[" CD "],},
            "GLY": {},
            "HIS": {" ND1":[" CG "," CE1"], " NE2":[" CD2"," CE1"],},
            "ILE": {},
            "LEU": {},
            "LYS": {" NZ ":[" CE "],},
            "MET": {" SD ":[" CG "," CE "],},
            "PHE": {},
            "PRO": {},
            "SER": {" OG ":[" CB "],},
            "THR": {" OG1":[" CB "],},
            "TRP": {" NE1":[" CD1"," CE2"],},
            "TYR": {" OH ":[" CZ "],},
            "VAL": {},
            "UNK": {},
            "MAS": {},
            " DA": {" N6 ":[" C6 ",], " N1 ":[" C2 "," C6 "], " N3 ":[" C2 "," C4 "], " N7 ":[" C5 "," C8 "],},
            " DG": {" N1 ":[" C2 "," C6 "], " N2 ":[" C2 ",], " N7 ":[" C5 "," C8 "], " O6 ":[" C6 ",], " N3 ":[" C2 "," C4 "], " N7 ":[" C5 "," C8 "],},
            " DC": {" N4 ":[" C4 ",], " N3 ":[" C2 "," C5 "], " O2 ":[" C2 ",],},
            " DT": {" N3 ":[" C2 "," C4 "], " O2 ":[" C2 ",], " O4 ":[" C4 ",],},
            " DX": {},
            " RA": {" O2'":[" C2'",],  " N6 ":[" C6 ",], " N1 ":[" C2 "," C6 "], " N3 ":[" C2 "," C4 "], " N7 ":[" C5 "," C8 "],},
            " RG": {" O2'":[" C2'",],  " N1 ":[" C2 "," C6 "], " N2 ":[" C2 ",], " N7 ":[" C5 "," C8 "], " O6 ":[" C6 ",], " N3 ":[" C2 "," C4 "], " N7 ":[" C5 "," C8 "],},
            " RC": {" O2'":[" C2'",],  " N4 ":[" C4 ",], " N3 ":[" C2 "," C5 "], " O2 ":[" C2 ",],},
            " RU": {" O2'":[" C2'",],  " N3 ":[" C2 "," C4 "], " O2 ":[" C2 ",], " O4 ":[" C4 ",],},
            " RX": {" O2'":[" C2'",], },
        }



        self.ideal_angle_dict = {
            'O': {
                1: 109.5*(np.pi/180),
                2: 180.0*(np.pi/180)},
            'N': {
                1: 120.0*(np.pi/180),
                2: 180.0*(np.pi/180)},
            'S': { # TO DO: CHECK IF BOND ANGLES ARE CORRECT!
                1: 109.5*(np.pi/180),
                2: 180.0*(np.pi/180)},
            'P': { # TO DO: CHECK IF BOND ANGLES ARE CORRECT!
                1: 120.0*(np.pi/180),
                2: 180.0*(np.pi/180)},
        }


    def _init_nuc_chemdata(self):

        self.nuc_resi_3letter = [" DA"," DG"," DC"," DT"," RA"," RG"," RC"," RU"]

        # Vectors between atom pairs that define each interaction edge of each base
        self.vec_atom_dict = {
            " DA": {"W_start":" N1 ","W_stop":" N6 ", "H_start":" N7 ","H_stop":" N6 ", "S_start":" C1'","S_stop":" N3 ", "B_start":" C1'","B_stop":" N9 " },
            " DG": {"W_start":" N1 ","W_stop":" O6 ", "H_start":" N7 ","H_stop":" O6 ", "S_start":" C1'","S_stop":" N3 ", "B_start":" C1'","B_stop":" N9 " },
            " DC": {"W_start":" N3 ","W_stop":" N4 ", "H_start":" C5 ","H_stop":" N4 ", "S_start":" C1'","S_stop":" O2 ", "B_start":" C1'","B_stop":" N1 " },
            " DT": {"W_start":" N3 ","W_stop":" O4 ", "H_start":" C5 ","H_stop":" O4 ", "S_start":" C1'","S_stop":" O2 ", "B_start":" C1'","B_stop":" N1 " },
            " RA": {"W_start":" N1 ","W_stop":" N6 ", "H_start":" N7 ","H_stop":" N6 ", "S_start":" C1'","S_stop":" N3 ", "B_start":" C1'","B_stop":" N9 " },
            " RG": {"W_start":" N1 ","W_stop":" O6 ", "H_start":" N7 ","H_stop":" O6 ", "S_start":" C1'","S_stop":" N3 ", "B_start":" C1'","B_stop":" N9 " },
            " RC": {"W_start":" N3 ","W_stop":" N4 ", "H_start":" C5 ","H_stop":" N4 ", "S_start":" C1'","S_stop":" O2 ", "B_start":" C1'","B_stop":" N1 " },
            " RU": {"W_start":" N3 ","W_stop":" O4 ", "H_start":" C5 ","H_stop":" O4 ", "S_start":" C1'","S_stop":" O2 ", "B_start":" C1'","B_stop":" N1 " },
        }

        self.vec_atom_inds = {s_i: {k_ij: HB_data.aa2long[HB_data.aa2num[s_i]].index(a_ij) for k_ij, a_ij in self.vec_atom_dict[s_i].items() } for s_i in self.nuc_resi_3letter}

        self.edge_to_ind = {'W':0 , 'H':1 , 'S':2 ,'B':3}
        self.ring_atom_list = [" N1 "," C2 "," N3 "," C4 "," C6 "," C5 "]
        self.ring_atom_inds = {s_i: [HB_data.aa2long[HB_data.aa2num[s_i]].index(a_ij)  for a_ij in self.ring_atom_list ] for s_i in self.nuc_resi_3letter}

def convert_mpnn_representation(S, X, X_m, rna_mask):
    """
    Given a sequence, atom coordinates, and atom mask in the NA-MPNN format,
    output the sequence, atom coordinates, and atom mask in an RFaa-like format.

    Arguments:
        S (np.int32 np.ndarray): an L length array representing the sequence of 
            the biomolecular assembly.
        X (np.float32 np.ndarray): an L x num_atom_types x 3 array representing
            the coordinates of each atom for each residue in the biomolecular
            assembly.
        X_m (np.int32 np.ndarray): an L x num_atom_types x 3 array mask that is
            1 if the corresponding atom in the specified residue was loaded
            and 0 otherwise.
        rna_mask (np.int32 np.ndarray): an L length array mask representing
            whether the residue is an RNA residue.
    
    Returns:
        S_rfaa (np.int32 np.ndarray): an L length array representing the 
            sequence of the biomolecular assembly in the RFaa format.
        X_rfaa (np.float32 np.ndarray): an L x num_atom_types x 3 array 
            representing the coordinates of each atom for each residue in the 
            biomolecular assembly in the RFaa format.
    """
    atom_idx_to_name = {atom_idx:atom_name for (atom_name, atom_idx) in pdb_dataset.atom_dict.items()}

    # Convert the sequence to the RFaa format, being aware of the possible
    # shared token representation of NA-MPNN.
    S_rfaa = []
    for i in range(S.shape[0]):
        restype_int = S[i]

        restype = pdb_dataset.int_to_restype[restype_int]

        if rna_mask[i]:
            # Handle the case when shared nucleic acid tokens are used. If 
            # shared tokens are not used, the RNA tokens still need to be
            # converted to the RFaa notation.
            if restype == "DA" or restype == "A":
                restype_rfaa = "RA"
            elif restype == "DC" or restype == "C":
                restype_rfaa = "RC"
            elif restype == "DG" or restype == "G":
                restype_rfaa = "RG"
            elif restype == "DT" or restype == "U":
                restype_rfaa = "RU"
            elif restype == "DX" or restype == "RX":
                restype_rfaa = "RX"
            else:
                raise Exception("RNA restype not recognized.")
        else:
            restype_rfaa = restype
            
        restype_int_rfaa = HB_data.aa2num_stripped[restype_rfaa]
        
        S_rfaa.append(restype_int_rfaa)
    
    S_rfaa = np.array(S_rfaa, dtype = np.int64)

    # Convert the atom coordinates to the RFaa format.
    X_rfaa = np.zeros((X.shape[0], HB_data.NTOTAL, 3), dtype = np.float32)
    for i in range(X.shape[0]):
        restype_int_rfaa = S_rfaa[i]
        for atom_idx in range(X.shape[1]):
            if X_m[i, atom_idx] == 1:
                atom_type = atom_idx_to_name[atom_idx]

                # Don't load any atoms beyond backbone for UNK, DX, RX.
                if (HB_data.num2aa[restype_int_rfaa] in ["UNK", " DX", " RX"]) and \
                   (atom_type not in HB_data.aa2long_stripped[restype_int_rfaa]):
                    continue

                # There are rare cases in the PDB where a DNA/RNA hybrid chain 
                # is mislabeled as DNA. In these cases, the data processing
                # pipeline labels RNA residues as DNA residues, and there is
                # an error with transfering the O2' atom into the RFaa format.
                if (HB_data.num2aa[restype_int_rfaa] in [" DA", " DC", " DG", " DT"]) and \
                     (atom_type == "O2'"):
                    continue

                # RFaa does not represent the OXT atom type.
                if atom_type == "OXT":
                    continue

                # Write the atom coordinates to the RFaa format.
                atom_idx_rfaa = HB_data.aa2long_stripped[restype_int_rfaa].index(atom_type)
                X_rfaa[i, atom_idx_rfaa] = X[i, atom_idx]
    
    return S_rfaa, X_rfaa

def get_base_pair_mask_and_index(S, X, X_m, rna_mask):
    """
    Given a sequence, atom coordinates, and atom mask, compute the base pairing
    residues and the canonical base pairing residues (represented as a mask
    and index of the base pairing partner).

    Arguments:
        S (np.int32 np.ndarray): an L length array representing the sequence of 
            the biomolecular assembly.
        X (np.float32 np.ndarray): an L x num_atom_types x 3 array representing
            the coordinates of each atom for each residue in the biomolecular
            assembly.
        X_m (np.int32 np.ndarray): an L x num_atom_types x 3 array mask that is
            1 if the corresponding atom in the specified residue was loaded
            and 0 otherwise.
        rna_mask (np.int32 np.ndarray): an L length array mask representing
            whether the residue is an RNA residue.
    
    Returns:
        base_pair_mask (np.int32 np.ndarray): an L length array mask that is
            1 if the corresponding residue is involved in a base pair
            interaction and 0 otherwise.
        base_pair_index (np.int64 np.ndarray): an L length array that
            represents the index of the partner residue in a base pairing
            interaction. For residues not in a base pairing interaction,
            defined as 0, but it is necessary to the base_pair_mask in
            conjunction.
        canonical_base_pair_mask (np.int32 np.ndarray): similar to 
            base_pair_mask, but limited to positions that make canonical base
            pairing interactions.
        canonical_base_pair_index (np.int64 np.ndarray): similar to
            base_pair_index, but limited to positions that make canonical base
            pairing interactions.
    """
    # Convert to the representation needed for the HB_data object.
    S_rfaa, X_rfaa = convert_mpnn_representation(S, X, X_m, rna_mask)

    hb_data = HB_data(torch.tensor(S_rfaa), 
                      torch.tensor(X_rfaa), 
                      compute_paired_bases=True,
                      compute_helical_params=True
                     )
    # basepairs_ij is only created if there is non-DX/RX nucleic acids in the
    # structure.
    if hb_data.is_na.sum() > 0:
        base_pairs_prob = hb_data.basepairs_ij.detach().cpu().numpy()
        base_pairs_binary = (base_pairs_prob > 0.5).astype(np.int32)

        # Only consider base pairing interactions that have one partner.
        base_pair_mask = (np.sum(base_pairs_binary, axis = -1) == 1).astype(np.int32)
        base_pair_index = np.argmax(base_pairs_binary, axis = -1).astype(np.int64)
    else:
        base_pair_mask = np.zeros(S_rfaa.shape[0], dtype = np.int32)
        base_pair_index = np.zeros(S_rfaa.shape[0], dtype = np.int64)

    # Base pair mask needs to be updated so that the base pairing partner
    # also exists.
    base_pair_mask = base_pair_mask * base_pair_mask[base_pair_index]

    # Make sure to update the base pair index using the base pair mask.
    base_pair_index = base_pair_index * base_pair_mask

    # Create the canonical base pair mask and index, removing any base pairing
    # interactions with non-canonical sequences.
    canonical_base_pair_mask = np.copy(base_pair_mask)
    canonical_base_pair_index = np.copy(base_pair_index)
    for i in range(len(S)):
        if base_pair_mask[i] == 1:
            restype_i = S[i]
            restype_j = S[base_pair_index[i]]
            if (restype_i, restype_j) not in pdb_dataset.na_canonical_base_pair_ints:
                canonical_base_pair_mask[i] = 0
                canonical_base_pair_mask[base_pair_index[i]] = 0
    
    # Make sure to update the canonical base pair index using the canonical base 
    # pair mask.
    canonical_base_pair_index = canonical_base_pair_index * canonical_base_pair_mask
    
    return base_pair_mask, base_pair_index, canonical_base_pair_mask, canonical_base_pair_index

# Get nearest neighbors
def get_nearest_interface_neighbors_to_res_i(X, protein_mask, na_mask, i, eps = 1E-6):
    if protein_mask[i] == 1:
        mask = na_mask
    elif na_mask[i] == 1:
        mask = protein_mask
    dX = X - X[i]
    D = mask * torch.sqrt(torch.sum(dX ** 2, 1) + eps)
    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - mask) * (D_max + eps)
    D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(num_neighbors, X.shape[0]), dim=-1, largest=False)
    return E_idx
    
def get_interface_masks(X, X_m, protein_mask, dna_mask, rna_mask):
    L = X.shape[0]
    na_mask = dna_mask + rna_mask #[L]

    interface_mask = np.zeros(L, dtype = np.int32)

    Ca = X[:,pdb_dataset.atom_dict["CA"],:]
    na_ref_atom = X[:,pdb_dataset.atom_dict[params["NA_REF_ATOM"]],:]

    side_chain_interface_mask = np.zeros(L, dtype = np.int32)
    nearest_protein_side_chain_index = np.zeros(L, dtype = np.int64)
    for i in range(L):
        nearest_neighbor_idx = get_nearest_interface_neighbors_to_res_i(torch.tensor(Ca + na_ref_atom), torch.tensor(protein_mask), torch.tensor(na_mask), i)
        nearest_protein_side_chain_distance = None
        for j in nearest_neighbor_idx:
            if not (na_mask[i] == 1 or na_mask[j] == 1):
                continue

            res_i_X = X[i] #[N,3]
            res_i_X_m = X_m[i] #[N]

            res_j_X = X[j] #[N,3]
            res_j_X_m = X_m[j] #[N]

            # Compute the per-atom pairwise distance.
            dX = res_i_X[:,None,:] - res_j_X[None,:,:] #[N,N,3]
            pairwise_atom_distances = np.sqrt(np.sum(dX ** 2, axis = -1)) #[N,N]

            # Mask out pairwise distances for atoms that do not exist.
            X_m_pairwise = res_i_X_m[:,None] * res_j_X_m[None,:] #[N,N]
            
            min_distance = np.min(pairwise_atom_distances[(X_m_pairwise == 1)])
            if min_distance < interface_distance_cutoff:
                if (protein_mask[i] == 1 and na_mask[j] == 1) or (protein_mask[j] == 1 and na_mask[i] == 1): 
                    interface_mask[i] = 1
                    interface_mask[j] = 1
            
            X_m_side_chain_pairwise = X_m_pairwise * side_chain_pairwise_mask

            if np.count_nonzero(X_m_side_chain_pairwise) > 0:
                min_side_chain_distance = np.min(pairwise_atom_distances[(X_m_side_chain_pairwise == 1)])
                if min_side_chain_distance < interface_distance_cutoff:
                    if (protein_mask[i] == 1 and na_mask[j] == 1) or (protein_mask[j] == 1 and na_mask[i] == 1): 
                        side_chain_interface_mask[i] = 1
                        side_chain_interface_mask[j] = 1
                    
                    if protein_mask[j] == 1 and \
                    (nearest_protein_side_chain_distance == None or \
                        min_side_chain_distance < nearest_protein_side_chain_distance):
                        nearest_protein_side_chain_index[i] = j
                        nearest_protein_side_chain_distance = min_side_chain_distance

    return interface_mask, side_chain_interface_mask, nearest_protein_side_chain_index

if __name__ == "__main__":
    # Load the command line arguments.
    input_csv_path = sys.argv[1]
    output_directory = sys.argv[2]
    modulo = int(sys.argv[3])
    remainder = int(sys.argv[4])

    # Load the csv, containing the structure path and pwm paths.
    df = pd.read_csv(input_csv_path)

    # Output directory file paths.
    sequences_directory = os.path.join(output_directory, "sequences")
    asmb_lengths_directory = os.path.join(output_directory, "asmb_lengths")
    asmb_interface_masks_directory = os.path.join(output_directory, "asmb_interface_masks")
    asmb_side_chain_interface_masks_directory = os.path.join(output_directory, "asmb_side_chain_interface_masks")
    asmb_nearest_protein_side_chain_index_directory = os.path.join(output_directory, "asmb_nearest_protein_side_chain_index")
    asmb_base_pair_masks_directory = os.path.join(output_directory, "asmb_base_pair_masks")
    asmb_base_pair_index_directory = os.path.join(output_directory, "asmb_base_pair_index")
    asmb_canonical_base_pair_masks_directory = os.path.join(output_directory, "asmb_canonical_base_pair_masks")
    asmb_canonical_base_pair_index_directory = os.path.join(output_directory, "asmb_canonical_base_pair_index")
    bad_directory = os.path.join(output_directory, "bad")

    # Make output directories.
    os.makedirs(sequences_directory, exist_ok = True)
    os.makedirs(asmb_lengths_directory, exist_ok = True)
    os.makedirs(asmb_interface_masks_directory, exist_ok = True)
    os.makedirs(asmb_side_chain_interface_masks_directory, exist_ok = True)
    os.makedirs(asmb_nearest_protein_side_chain_index_directory, exist_ok = True)
    os.makedirs(asmb_base_pair_masks_directory, exist_ok = True)
    os.makedirs(asmb_base_pair_index_directory, exist_ok = True)
    os.makedirs(asmb_canonical_base_pair_masks_directory, exist_ok = True)
    os.makedirs(asmb_canonical_base_pair_index_directory, exist_ok = True)
    os.makedirs(bad_directory, exist_ok = True)

    # Preprocess data.
    for iii in range(len(df)):
        if (iii + 1) % modulo != remainder:
            continue
        
        example_dict = df.iloc[iii].to_dict()

        structure_file_name = os.path.basename(example_dict["structure_path"])
        # Handle GZipped files.
        if structure_file_name.endswith(".gz"):
            structure_name = os.path.splitext(os.path.splitext(structure_file_name)[0])[0]
        else:
            structure_name = os.path.splitext(structure_file_name)[0]
        
        sequences_path = os.path.join(sequences_directory, structure_name + ".csv")
        asmb_lengths_path = os.path.join(asmb_lengths_directory, structure_name + ".npy")
        asmb_interface_masks_path = os.path.join(asmb_interface_masks_directory, structure_name + ".npy")
        asmb_side_chain_interface_masks_path = os.path.join(asmb_side_chain_interface_masks_directory, structure_name + ".npy")
        asmb_nearest_protein_side_chain_index_path = os.path.join(asmb_nearest_protein_side_chain_index_directory, structure_name + ".npy")
        asmb_base_pair_masks_path = os.path.join(asmb_base_pair_masks_directory, structure_name + ".npy")
        asmb_base_pair_index_path = os.path.join(asmb_base_pair_index_directory, structure_name + ".npy")
        asmb_canonical_base_pair_masks_path = os.path.join(asmb_canonical_base_pair_masks_directory, structure_name + ".npy")
        asmb_canonical_base_pair_index_path = os.path.join(asmb_canonical_base_pair_index_directory, structure_name + ".npy")
        bad_path = os.path.join(bad_directory, structure_name + ".txt")

        try:
            assemblies, chain_sequences = pdb_dataset.load_for_structure_preprocessing(example_dict)
        except Exception as e:
            write_text_file(bad_path, str(e))
            continue

        if assemblies == "pass" or (len(assemblies) == 0):
            write_text_file(bad_path, "cifutils_failed_to_load_assemblies")
            continue

        asmb_lengths_dict = {}
        asmb_interface_masks_dict = {}
        asmb_side_chain_interface_masks_dict = {}
        asmb_nearest_protein_side_chain_index_dict = {}
        asmb_base_pair_masks_dict = {}
        asmb_base_pair_index_dict = {}
        asmb_canonical_base_pair_masks_dict = {}
        asmb_canonical_base_pair_index_dict = {}
        missing_na_count = 0
        for (assembly_id, out_dict) in assemblies:
            # Filter out assemblies with no resolved/occupied nucleic acids.
            if (out_dict["dna_L"] == 0) and (out_dict["rna_L"] == 0):
                missing_na_count += 1
                continue

            # Get the base pair mask and index. If the sequence longer than the
            # normal batch size for MPNN, then the base pair mask and index will 
            # be empty.
            if out_dict["S"].shape[0] > residue_cutoff:
                base_pair_mask = np.zeros(out_dict["S"].shape, dtype = np.int32)
                base_pair_index = np.zeros(out_dict["S"].shape, dtype = np.int64)
                canonical_base_pair_mask = np.zeros(out_dict["S"].shape, dtype = np.int32)
                canonical_base_pair_index = np.zeros(out_dict["S"].shape, dtype = np.int64)
            else:
                base_pair_mask, base_pair_index, canonical_base_pair_mask, canonical_base_pair_index = \
                    get_base_pair_mask_and_index(out_dict["S"], 
                                                    out_dict["X"], 
                                                    out_dict["X_m"], 
                                                    out_dict["rna_mask"])
            
            # Get the interface masks.
            interface_mask, side_chain_interface_mask, nearest_protein_side_chain_index = \
                get_interface_masks(out_dict["X"], 
                                    out_dict["X_m"], 
                                    out_dict["protein_mask"], 
                                    out_dict["dna_mask"], 
                                    out_dict["rna_mask"])
            
            # Save the per-assembly data.
            asmb_lengths_dict[assembly_id] = (out_dict["macromolecule_L"], out_dict["protein_L"], out_dict["dna_L"], out_dict["rna_L"])
            asmb_interface_masks_dict[assembly_id] = interface_mask
            asmb_side_chain_interface_masks_dict[assembly_id] = side_chain_interface_mask
            asmb_nearest_protein_side_chain_index_dict[assembly_id] = nearest_protein_side_chain_index
            asmb_base_pair_masks_dict[assembly_id] = base_pair_mask
            asmb_base_pair_index_dict[assembly_id] = base_pair_index
            asmb_canonical_base_pair_masks_dict[assembly_id] = canonical_base_pair_mask
            asmb_canonical_base_pair_index_dict[assembly_id] = canonical_base_pair_index

        if len(list(asmb_lengths_dict)) > 0:
            chain_sequences_lines = ["chain_id,chain_type,sequence"]
            for chain_sequence_line in chain_sequences:
                chain_sequence_line = tuple(map(lambda x: "" if x is None else x, chain_sequence_line))
                chain_sequences_lines.append(",".join(chain_sequence_line))
            chain_sequences_str = "\n".join(chain_sequences_lines)
            write_text_file(sequences_path, chain_sequences_str)

            np.save(asmb_lengths_path, asmb_lengths_dict)
            np.save(asmb_interface_masks_path, asmb_interface_masks_dict)
            np.save(asmb_side_chain_interface_masks_path, asmb_side_chain_interface_masks_dict)
            np.save(asmb_nearest_protein_side_chain_index_path, asmb_nearest_protein_side_chain_index_dict)
            np.save(asmb_base_pair_masks_path, asmb_base_pair_masks_dict)
            np.save(asmb_base_pair_index_path, asmb_base_pair_index_dict)
            np.save(asmb_canonical_base_pair_masks_path, asmb_canonical_base_pair_masks_dict)
            np.save(asmb_canonical_base_pair_index_path, asmb_canonical_base_pair_index_dict)
        elif missing_na_count == len(assemblies):
            write_text_file(bad_path, "all_assemblies_no_resolved_and_occupied_nucleic_acids")
            continue
        else:
            write_text_file(bad_path, "all_assemblies_failed")
            continue