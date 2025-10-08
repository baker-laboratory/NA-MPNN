import sys
import numpy as np
import torch
from prody import *
confProDy(verbosity='none')

def make_pair_bias(chain_labels, R_idx, pair_bias_AA):
    d_chains = ((chain_labels[:, None] - chain_labels[None,:])==0).long()
    upper_diag = (R_idx[1:]-R_idx[:-1]==1).long()
    lower_diag = (R_idx[:-1]-R_idx[1:]==-1).long()
    u_mask = torch.diag(upper_diag, 1)
    u_mask = u_mask*d_chains
    l_mask = torch.diag(lower_diag, -1)
    l_mask = l_mask*d_chains
    pair_bias = (u_mask[None,:,None,:,None]*pair_bias_AA[None,None,:,None,:]+l_mask[None,:,None,:,None]*torch.transpose(pair_bias_AA, 0, 1)[None,None,:,None,:]) #[1,L,21,L,21]
    return pair_bias

def get_seq_rec(S: torch.Tensor, 
                S_pred: torch.Tensor, 
                mask: torch.Tensor):
    """
    S : true sequence shape=[batch, length]
    S_pred : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    average : averaged sequence recovery shape=[batch]
    """
    match = (S == S_pred)
    average = torch.sum(match*mask, dim=-1)/torch.sum(mask, dim=-1)
    return average

def write_text_file(path, contents):
    with open(path, mode = "wt") as f:
        f.write(contents)
    
def get_score(S: torch.Tensor, 
              log_probs: torch.Tensor, 
              mask: torch.Tensor,
              num_letters: int):
    """
    S : true sequence shape=[batch, length]
    log_probs : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]
    num_letters : the number of letters in the output alphabet.

    average_loss : averaged categorical cross entropy (CCE) [batch]
    loss_per_resdue : per position CCE [batch, length]
    """
    S_one_hot = torch.nn.functional.one_hot(S, num_letters)
    loss_per_residue = -(S_one_hot*log_probs).sum(-1) #[B, L]
    average_loss = torch.sum(loss_per_residue*mask, dim=-1)/(torch.sum(mask, dim=-1)+1e-8)
    return average_loss, loss_per_residue

def get_aligned_coordinates(macromolecule_atoms, 
                            reference_atom_dict: dict, 
                            atom_name: str):
    """
    macromolecule_atoms: prody atom group
    reference_atom_dict: mapping between chain_residue_idx_icodes and integers
    atom_name: atom to be parsed; e.g. CA
    """
    atom_atoms = macromolecule_atoms.select(f'name {atom_name}')
    
    if atom_atoms != None:
        atom_coords = atom_atoms.getCoords()
        atom_resnums = atom_atoms.getResnums()
        atom_chain_ids = atom_atoms.getChids()
        atom_icodes = atom_atoms.getIcodes()
    
    atom_coords_= np.zeros([len(reference_atom_dict),3], np.float32)
    atom_coords_m= np.zeros([len(reference_atom_dict)], np.int32)
    if atom_atoms != None:
        for i in range(len(atom_resnums)):
            code=atom_chain_ids[i]+"_"+str(atom_resnums[i])+"_"+atom_icodes[i]
            if code in list(reference_atom_dict):
                atom_coords_[reference_atom_dict[code],:] = atom_coords[i]
                atom_coords_m[reference_atom_dict[code]] = 1
    return atom_coords_, atom_coords_m

def read_text_file(path):
    with open(path, mode="rt") as f:
        return f.read()

def parse_PDB(input_path: str, 
              device: str="cpu", 
              chains: list=[],
              parse_all_atoms: bool=False,
              model_type: str = "protein_mpnn",
              parse_na_only=False,
              na_shared_tokens=False,
              load_residues_with_missing_atoms=0
             ):
    """
    input_path : path for the input PDB
    device: device for the torch.Tensor
    chains: a list specifying which chains need to be parsed; e.g. ["A", "B"]
    parse_all_atoms: if False parse only N,CA,C,O otherwise all 37 atoms
    model_type : string representing the model type
    """
    element_list = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mb', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']
    element_list = [item.upper() for item in element_list]
    element_dict = dict(zip(element_list, range(1,len(element_list))))

    protein_restypes = [
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

    dna_restypes = [
        'DA',
        'DC',
        'DG',
        'DT',
        'DX'
    ]

    rna_restypes = [
        'A',
        'C',
        'G',
        'U',
        'RX'
    ]

    polytypes = [
        'PP',
        'DNA',
        'RNA',
        'UNK',
        'MAS',
        'PAD'
    ]

    polytype_to_int = dict(zip(polytypes, range(len(polytypes))))

    if not parse_all_atoms:
        atom_types = ['N', 'CA', 'C', 'O', #protein atoms
                      'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'" #nucleic acid atoms
                     ]
    else:
        atom_types = ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT', #protein atoms
                      'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N9', 'C8', 'C7', 'N7', 'C6', 'N6', 'O6', 'C5', 'C4', 'N4', 'O4', 'N3', 'C2', 'N2', 'O2', 'N1' #nucleic acid atoms
                     ]

    atom_order = dict(zip(atom_types, range(len(atom_types))))

    protein_backbone_atoms = ["N", "CA", "C", "O"]
    dna_backbone_atoms = ['OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]
    rna_backbone_atoms = ['OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
    na_backbone_atoms = ['OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]

    protein_backbone_indices = list(map(lambda x: atom_order[x], protein_backbone_atoms))
    dna_backbone_indices = list(map(lambda x: atom_order[x], dna_backbone_atoms))
    rna_backbone_indices = list(map(lambda x: atom_order[x], rna_backbone_atoms))

    backbone_atom_indices = list(map(lambda x: atom_order[x], protein_backbone_atoms + na_backbone_atoms))

    macromolecule_backbone_atoms = {"protein": protein_backbone_atoms, 
                                    "nucleic": na_backbone_atoms}
    macromolecule_reference_atoms = {"protein": "CA", 
                                     "nucleic": "C1'"}

    if model_type == "na_mpnn":
        restypes = [
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

        # residue to int ordering is different than other models. This matches the training code for na_mpnn.
        restype_to_int = dict(zip(restypes, range(len(restypes))))

        if na_shared_tokens:
            restype_to_int["A"] = restype_to_int["DA"]
            restype_to_int["C"] = restype_to_int["DC"]
            restype_to_int["G"] = restype_to_int["DG"]
            restype_to_int["U"] = restype_to_int["DT"]
            restype_to_int["RX"] = restype_to_int["DX"]

        macromolecule_types_to_include = ["protein", "nucleic"]
    else:
        print("Choose --model_type flag from currently available models")
        sys.exit()

    atoms = parsePDB(input_path)
    atoms = atoms.select('occupancy > 0')
    if chains:
        str_out=""
        for item in chains:
            str_out+=" chain "+item+" or"
        atoms=atoms.select(str_out[1:-3])
    
    if parse_na_only:
        atoms = atoms.select("nucleic")

    macromolecule_atoms = atoms.select(" or ".join(macromolecule_types_to_include))

    backbone_select_strings = []
    for macromolecule_type in macromolecule_types_to_include:
        backbone_atoms = macromolecule_backbone_atoms[macromolecule_type]
        tmp_select_str = "(" + macromolecule_type + " and ("
        tmp_select_str += "name " + " or name ".join(backbone_atoms)
        tmp_select_str += "))"
        backbone_select_strings.append(tmp_select_str)

    backbone = atoms.select(" or ".join(backbone_select_strings))
    other_atoms = atoms.select("not " + " and not ".join(macromolecule_types_to_include + ["water"]))
    water_atoms = atoms.select("water")

    reference_select_strings = []
    for macromolecule_type in macromolecule_types_to_include:
        reference_atom_name = macromolecule_reference_atoms[macromolecule_type]
        tmp_select_str = "(" + macromolecule_type + " and name " + reference_atom_name + ")"
        reference_select_strings.append(tmp_select_str)
    
    # CA for proteins, C1' for nucleic
    reference_atoms = macromolecule_atoms.select(" or ".join(reference_select_strings))
    reference_resnums = reference_atoms.getResnums()
    reference_chain_ids = reference_atoms.getChids()
    reference_icodes = reference_atoms.getIcodes()

    na_reference_atoms = macromolecule_atoms.select("nucleic and name C1'")

    reference_atom_dict={}
    for i in range(len(reference_resnums)):
        code=reference_chain_ids[i]+"_"+str(reference_resnums[i])+"_"+reference_icodes[i]
        reference_atom_dict[code] = i

    xyz_65 = np.zeros([len(reference_atom_dict), 65, 3], np.float32)
    xyz_65_m = np.zeros([len(reference_atom_dict), 65], np.int32)
    for atom_name in atom_types:
        xyz, xyz_m = get_aligned_coordinates(macromolecule_atoms, reference_atom_dict, atom_name)
        xyz_65[:,atom_order[atom_name], :] = xyz
        xyz_65_m[:,atom_order[atom_name]] = xyz_m

    X = xyz_65[:, backbone_atom_indices]
    X_m = xyz_65_m[:, backbone_atom_indices]

    N = xyz_65[:,atom_order["N"],:]
    CA = xyz_65[:,atom_order["CA"],:]
    C = xyz_65[:,atom_order["C"],:]
    O = xyz_65[:,atom_order["O"],:]

    b = CA - N
    c = C - CA
    a = np.cross(b, c, axis=-1)
    CB = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA

    chain_labels = np.array(reference_atoms.getChindices(), dtype=np.int32)
    R_idx = np.array(reference_resnums, dtype=np.int32)
    S = reference_atoms.getResnames()
    
    if load_residues_with_missing_atoms:
        protein_mask = np.zeros_like(S, dtype = np.int32)
        dna_mask = np.zeros_like(S, dtype = np.int32)
        rna_mask = np.zeros_like(S, dtype = np.int32)
        for i, resname in enumerate(S):
            if resname in protein_restypes:
                protein_mask[i] = 1
            elif resname in dna_restypes:
                dna_mask[i] = 1
            elif resname in rna_restypes:
                rna_mask[i] = 1
    else:
        protein_mask = np.prod(xyz_65_m[:, protein_backbone_indices], axis = -1)
        rna_mask = np.prod(xyz_65_m[:, rna_backbone_indices], axis = -1)
        # When creating the polymer masks this way, it is necessary to subtract the
        # rna mask, since the rna will also have all the dna backbone atoms.
        dna_mask = np.prod(xyz_65_m[:, dna_backbone_indices], axis = -1) - rna_mask
    
    rna_mask_for_token_conversion = xyz_65_m[:, atom_order["O2'"]]
    
    mask = protein_mask + dna_mask + rna_mask

    R_polymer_type = protein_mask * polytype_to_int["PP"] + \
            dna_mask * polytype_to_int["DNA"] + \
            rna_mask * polytype_to_int["RNA"] + \
            (1 - protein_mask - dna_mask - rna_mask) * polytype_to_int["UNK"]

    if model_type == "na_mpnn":
        S_int = []
        for i, AA in enumerate(list(S)):
            if protein_mask[i] == 1:
                unknown_token = "UNK"
            elif dna_mask[i] == 1:
                unknown_token = "DX"
            elif rna_mask[i] == 1:
                unknown_token = "RX"
            else:
                unknown_token = "UNK"
            S_int.append(restype_to_int.get(AA, restype_to_int[unknown_token]))
        S = np.array(S_int, np.int32)
    else:
        print("Choose --model_type flag from currently available models")
        sys.exit()
    
    try: 
        Y = np.array(other_atoms.getCoords(), dtype=np.float32)
        Y_t = list(other_atoms.getElements())
        Y_t = np.array([element_dict[y_t.upper()] if y_t.upper() in element_list else 0 for y_t in Y_t], dtype=np.int32)
        Y_m = (Y_t != 1)*(Y_t != 0)

        Y = Y[Y_m,:]
        Y_t = Y_t[Y_m]
        Y_m = Y_m[Y_m]
    except:
        Y = np.zeros([1, 3], np.float32)
        Y_t = np.zeros([1], np.int32)
        Y_m = np.zeros([1], np.int32)

    output_dict = {}
    output_dict['X'] = torch.tensor(X, device=device, dtype=torch.float32)
    output_dict["X_m"] = torch.tensor(X_m, device=device, dtype=torch.int32)
    output_dict['mask'] = torch.tensor(mask, device=device, dtype=torch.int32)
    output_dict['Y'] = torch.tensor(Y, device=device, dtype=torch.float32)
    output_dict['Y_t'] = torch.tensor(Y_t, device=device, dtype=torch.int32)
    output_dict['Y_m'] = torch.tensor(Y_m, device=device, dtype=torch.int32)

    output_dict['R_idx'] = torch.tensor(R_idx, device=device, dtype=torch.int32)
    output_dict['chain_labels'] = torch.tensor(chain_labels, device=device, dtype=torch.int32)

    output_dict["chain_letters"] = list(np.array(reference_chain_ids))

    na_chain_ids = []
    if na_reference_atoms is None:
        na_chain_ids = np.array([])
    else:
        for i, chain_id in enumerate(list(reference_chain_ids)):
            if dna_mask[i] or rna_mask[i]:
                na_chain_ids.append(chain_id)

    output_dict["na_chain_letters"] = na_chain_ids

    output_dict['protein_mask'] = torch.tensor(protein_mask, device=device, dtype=torch.int32)
    output_dict['dna_mask'] = torch.tensor(dna_mask, device=device, dtype=torch.int32)
    output_dict['rna_mask'] = torch.tensor(rna_mask, device=device, dtype=torch.int32)

    output_dict['rna_mask_for_token_conversion'] = torch.tensor(rna_mask_for_token_conversion, device=device, dtype=torch.int32)

    output_dict['R_polymer_type'] = torch.tensor(R_polymer_type, device=device, dtype=torch.int64)

    output_dict['S'] = torch.tensor(S, device=device, dtype=torch.int32)

    output_dict["xyz_65"] = torch.tensor(xyz_65, device=device, dtype=torch.float32)
    output_dict["xyz_65_m"] = torch.tensor(xyz_65_m, device=device, dtype=torch.int32)
    
    mask_c = []
    chain_list = list(set(output_dict["chain_letters"]))
    chain_list.sort()
    for chain in chain_list:
        mask_c.append(torch.tensor([chain==item for item in output_dict["chain_letters"]], device=device, dtype=bool))

    output_dict["mask_c"] = mask_c
    output_dict["chain_list"] = chain_list
    
    return output_dict, backbone, other_atoms, reference_icodes, water_atoms

def featurize(input_dict):
    output_dict = {}
    R_idx_list = []
    count=0
    R_idx_prev=-100000
    for R_idx in list(input_dict["R_idx"]):
        if R_idx_prev==R_idx:
            count+=1
        R_idx_list.append(R_idx+count)
        R_idx_prev=R_idx
    R_idx_renumbered = torch.tensor(R_idx_list, device=R_idx.device)
    output_dict["R_idx"] = R_idx_renumbered[None,]
    output_dict["R_idx_original"] = input_dict["R_idx"][None,]
    output_dict["chain_labels"] = input_dict["chain_labels"][None,]
    output_dict["S"] = input_dict["S"][None,]
    output_dict["chain_mask"] = input_dict["chain_mask"][None,]
    output_dict["mask"] = input_dict["mask"][None,]

    output_dict['protein_mask'] = input_dict['protein_mask'][None,]
    output_dict['dna_mask'] = input_dict['dna_mask'][None,]
    output_dict['rna_mask'] = input_dict['rna_mask'][None,]

    output_dict['rna_mask_for_token_conversion'] = input_dict['rna_mask_for_token_conversion'][None,]

    output_dict['R_polymer_type'] = input_dict['R_polymer_type'][None,]

    output_dict["X"] = input_dict["X"][None,]
    output_dict["X_m"] = input_dict["X_m"][None,]

    output_dict["xyz_65"] = input_dict["xyz_65"][None,]
    output_dict["xyz_65_m"] = input_dict["xyz_65_m"][None,]

    return output_dict
