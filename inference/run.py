import argparse
import sys

def main(args):
    from prody import writePDB
    import torch
    import random
    import json
    import numpy as np
    import os.path
    from data_utils import make_pair_bias, get_seq_rec, get_score, parse_PDB, featurize
    from model_utils import ProteinMPNN

    if args.model_type == "na_mpnn":
        atom_types = [
            'N', 'CA', 'C', 'O', #protein atoms
            'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'" #nucleic acid atoms
        ]
        atom_dict = dict(zip(atom_types, range(len(atom_types))))

        polytypes = [
            'PP',
            'DNA',
            'RNA',
            'UNK',
            'MAS',
            'PAD'
        ]

        polytype_to_int = dict(zip(polytypes, range(len(polytypes))))

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

        restype_3_to_1 = {
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

        restype_to_int = dict(zip(restypes, range(len(restypes))))
        int_to_restype = dict(zip(range(len(restypes)), restypes))

        alphabet = [restype_3_to_1[int_to_restype[i]] for i in range(len(int_to_restype))]

        restype_1to3 = {v: k for k, v in restype_3_to_1.items()}

        dna_char_to_rna_char = dict()
        if args.na_shared_tokens:
            restype_to_int["A"] = restype_to_int["DA"]
            restype_to_int["C"] = restype_to_int["DC"]
            restype_to_int["G"] = restype_to_int["DG"]
            restype_to_int["U"] = restype_to_int["DT"]
            restype_to_int["RX"] = restype_to_int["DX"]

            dna_char_to_rna_char[restype_3_to_1["DA"]] = restype_3_to_1["A"]
            dna_char_to_rna_char[restype_3_to_1["DC"]] = restype_3_to_1["C"]
            dna_char_to_rna_char[restype_3_to_1["DG"]] = restype_3_to_1["G"]
            dna_char_to_rna_char[restype_3_to_1["DT"]] = restype_3_to_1["U"]
            dna_char_to_rna_char[restype_3_to_1["DX"]] = restype_3_to_1["RX"]

        restype_STRtoINT = {restype_3_to_1[k]: v for k, v in restype_to_int.items()}
        restype_INTtoSTR = dict()
        for k, v in restype_STRtoINT.items():
            if v not in restype_INTtoSTR:
                restype_INTtoSTR[v] = k

        vocab = 33
        num_letters = 33
    else:
        print("Choose --model_type flag from currently available models")
        sys.exit()
    
    #fix seeds
    if args.seed:
        seed=args.seed
    else:
        seed=int(np.random.randint(0, high=99999, size=1, dtype=int)[0])

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    #----

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    #make folders for outputs
    folder_for_outputs = args.out_folder
    base_folder = folder_for_outputs
    if base_folder[-1] != '/':
        base_folder = base_folder + '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)
    if not os.path.exists(base_folder + 'seqs') and args.output_sequences:
        os.makedirs(base_folder + 'seqs', exist_ok=True)
    if not os.path.exists(base_folder + 'backbones') and args.output_pdbs:
        os.makedirs(base_folder + 'backbones', exist_ok=True)
    if not os.path.exists(base_folder + 'specificity') and args.output_specificity:
        os.makedirs(base_folder + 'specificity', exist_ok=True)
    if args.save_stats:
        if not os.path.exists(base_folder + 'stats'):
            os.makedirs(base_folder + 'stats', exist_ok=True)
    #----

    if args.model_type == "na_mpnn":
        checkpoint_path = args.checkpoint_na_mpnn
    else:
        print("Choose --model_type flag from currently available models")
        sys.exit()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if args.model_type == "na_mpnn":
        k_neighbors = 32
    else:
        print("Choose --model_type flag from currently available models")
        sys.exit()

    if args.k_neighbors is not None:
        k_neighbors = args.k_neighbors

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        model_type=args.model_type,
        vocab=vocab,
        num_letters=num_letters,
        atom_dict=atom_dict,
        restype_to_int=restype_to_int,
        polytype_to_int=polytype_to_int
    )

    #load pretrained parameters
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    #----

    #make amino acid bias array [num_letters]
    bias_AA = torch.zeros([num_letters], device=device, dtype=torch.float32)
    if args.bias_AA:
        tmp = [item.split(":") for item in args.bias_AA.split(",")]
        a1 = [b[0] for b in tmp]
        a2 = [float(b[1]) for b in tmp]
        for i, AA in enumerate(a1):
            bias_AA[restype_STRtoINT[AA]] = a2[i]
    #----

    #make amino acid pair bias array [num_letters, num_letters]
    pair_bias_AA = torch.zeros([num_letters,num_letters], dtype=torch.float32, device=device)
    if args.pair_bias_AA:
        tmp = [item.split(":") for item in args.pair_bias_AA.split(",")]
        a1 = [b[0][0] for b in tmp]
        a2 = [b[0][1] for b in tmp]
        a3 = [float(b[1]) for b in tmp]
        for i, AA in enumerate(a1):
            pair_bias_AA[restype_STRtoINT[AA], restype_STRtoINT[a2[i]]] = a3[i]
    #----
      
    #make array to indicate which amino acids need to be omitted [num_letters]
    omit_AA_list = args.omit_AA

    #if using shared DNA and RNA tokens, omit the separate RNA (legacy) tokens.
    if args.na_shared_tokens:
        omit_AA_list = omit_AA_list + "bdhuy"

    omit_AA = torch.tensor(np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32), device=device)
    #----

    if args.fixed_pos_by_pdb:
        with open(args.fixed_pos_by_pdb, 'r') as fh:
            fixed_pos_by_pdb = json.load(fh)
    else:
        fixed_residues = [item for item in args.fixed_residues.split()]
        fixed_pos_by_pdb = {
            args.pdb_path: fixed_residues
        }

    for pdb, fixed_residues in fixed_pos_by_pdb.items():
        #adjust input PDB name by dropping .pdb if it does exist
        name = pdb[pdb.rfind("/")+1:]
        if name[-4:] == ".pdb":
            name = name[:-4]
        elif name[-4:] == ".cif":
            name = name[:-4]
        #----
        if "pdb" == name[:3]:
            pdb_id = name[3:]
        else:
            pdb_id = name

        #parse PDB file
        macromolecule_dict, backbone, other_atoms, icodes, water_atoms = \
            parse_PDB(
                pdb,
                device=device, 
                chains=args.parse_these_chains_only,
                model_type=args.model_type,
                parse_na_only=args.parse_na_only,
                na_shared_tokens=args.na_shared_tokens,
                load_residues_with_missing_atoms=args.load_residues_with_missing_atoms
            )
        #----
        
        #make chain_letter + residue_idx + insertion_code mapping to integers
        R_idx_list = list(macromolecule_dict["R_idx"].cpu().numpy())
        chain_letters_list = list(macromolecule_dict["chain_letters"])
        encoded_residues = []
        for i in range(len(R_idx_list)):
            tmp = str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
            encoded_residues.append(tmp)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        #----

        #make fixed positions array; those residues will be kept fixed
        fixed_positions = torch.tensor([int(item not in fixed_residues) for item in encoded_residues], device=device)
        #----

        #specify which residues need to be designed; everything else will be fixed
        if args.redesigned_residues:
            redesigned_residues = [item for item in args.redesigned_residues.split()]
            redesigned_positions = torch.tensor([int(item not in redesigned_residues) for item in encoded_residues], device=device)
        else:
            redesigned_positions = torch.zeros_like(fixed_positions)
        #----

        #specify which chains need to be redesigned
        if type(args.chains_to_design) == str:
            chains_to_design_list = args.chains_to_design.split(",")
        else:
            chains_to_design_list = macromolecule_dict["chain_letters"]

        if args.design_na_only:
            tmp_chains_to_design_list = []
            for letter in chains_to_design_list:
                if letter in macromolecule_dict["na_chain_letters"]:
                    tmp_chains_to_design_list.append(letter)
            chains_to_design_list = tmp_chains_to_design_list

        chain_mask = torch.tensor(np.array([item in chains_to_design_list for item in macromolecule_dict["chain_letters"]],dtype=np.int32), device=device)
        #----

        #create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
        macromolecule_dict["chain_mask"] = chain_mask*fixed_positions*(1-redesigned_positions)
        #----

        #specify which residues are linked
        if args.symmetry_residues:
            symmetry_residues_list_of_lists = [x.split(',') for x in args.symmetry_residues.split('|')]
            remapped_symmetry_residues=[]
            for t_list in symmetry_residues_list_of_lists:
                tmp_list=[]
                for t in t_list:
                    tmp_list.append(encoded_residue_dict[t])
                remapped_symmetry_residues.append(tmp_list) 
        else:
            remapped_symmetry_residues=[[]]
        #----

        #specify linking weights
        if args.symmetry_weights:
            symmetry_weights = [[float(item) for item in x.split(',')] for x in args.symmetry_weights.split('|')]
        elif not args.symmetry_weights and args.symmetry_residues:
            symmetry_weights = [[float(1.0) for item in x.split(',')] for x in args.symmetry_residues.split('|')]
        else:
            symmetry_weights = [[]]
        #----

        #set other atom bfactors to 0.0
        if other_atoms:
            other_bfactors = other_atoms.getBetas()
            other_atoms.setBetas(other_bfactors*0.0)
        #----
        
        with torch.no_grad():
            #run featurize to remap R_idx and add batch dimension
            feature_dict = featurize(macromolecule_dict)
            feature_dict["batch_size"] = args.batch_size
            B, L, _, _ = feature_dict["X"].shape #batch size should be 1 for now.
            #----

            #add additional keys to the feature dictionary
            feature_dict["temperature"] = args.temperature
            feature_dict["bias"] = (-1e8*omit_AA[None,None,:]+bias_AA).repeat([1,L,1])
            if args.pair_bias_AA:
                feature_dict["pair_bias"] = make_pair_bias(feature_dict["chain_labels"][0], feature_dict["R_idx"][0], pair_bias_AA)
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights
            #----

            sampling_probs_list = []
            log_probs_list = []
            decoding_order_list = []
            S_list = []
            loss_list = []
            loss_per_residue_list = []
            loss_XY_list = []
            for _ in range(args.number_of_batches):
                feature_dict["randn"] = torch.randn([feature_dict["batch_size"], feature_dict["mask"].shape[1]], device=device)
                #main step-----
                output_dict = model.sample(feature_dict)

                #compute confidence scores
                loss, loss_per_residue = get_score(output_dict["S"], output_dict["log_probs"], feature_dict["mask"]*feature_dict["chain_mask"], num_letters)
                combined_mask = feature_dict["mask"]*feature_dict["chain_mask"]
                loss_XY, _ = get_score(output_dict["S"], output_dict["log_probs"], combined_mask, num_letters)
                #-----
                S_list.append(output_dict["S"])
                log_probs_list.append(output_dict["log_probs"])
                sampling_probs_list.append(output_dict["sampling_probs"])
                decoding_order_list.append(output_dict["decoding_order"])
                loss_list.append(loss)
                loss_per_residue_list.append(loss_per_residue)
                loss_XY_list.append(loss_XY)

            S_stack = torch.cat(S_list, 0)
            log_probs_stack = torch.cat(log_probs_list, 0)
            sampling_probs_stack = torch.cat(sampling_probs_list, 0)
            decoding_order_stack = torch.cat(decoding_order_list, 0)
            loss_stack = torch.cat(loss_list, 0)
            loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
            loss_XY_stack = torch.cat(loss_XY_list, 0)
            rec_mask = feature_dict["mask"][:1]*feature_dict["chain_mask"][:1]
            rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)
            
            #make input sequence string separated by / between different chains
            native_seq_list = []
            for i, AA in enumerate(feature_dict["S"][0].cpu().numpy()):
                if feature_dict["rna_mask_for_token_conversion"][0, i] == 1:
                    native_seq_list.append(dna_char_to_rna_char.get(restype_INTtoSTR[AA], restype_INTtoSTR[AA]))
                else:
                    native_seq_list.append(restype_INTtoSTR[AA])
            native_seq = "".join(native_seq_list)
            seq_np = np.array(list(native_seq))
            seq_out_str = []
            for mask in macromolecule_dict['mask_c']:
                seq_out_str += list(seq_np[mask.cpu().numpy()])
                seq_out_str += ['/']
            seq_out_str = "".join(seq_out_str)[:-1]
            #------

            output_fasta = base_folder + '/seqs/' + name + '.fa' + args.file_ending
            output_backbones = base_folder + '/backbones/'
            output_stats_path = base_folder + 'stats/' + name + ".pt"
            output_specificity = os.path.join(base_folder, "specificity")

            out_dict = {}
            out_dict["generated_sequences"] = S_stack.cpu()
            out_dict["sampling_probs"] = sampling_probs_stack.cpu()
            out_dict["log_probs"] = log_probs_stack.cpu()
            out_dict["decoding_order"] = decoding_order_stack.cpu()
            out_dict["native_sequence"] = feature_dict["S"][0].cpu()
            out_dict["mask"] = feature_dict["mask"][0].cpu()
            out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
            out_dict["seed"] = seed
            out_dict["temperature"] = args.temperature
            if args.save_stats:
                torch.save(out_dict, output_stats_path)

            if args.output_specificity:
                predicted_ppm = np.mean(sampling_probs_stack.cpu().numpy().astype(np.float64), axis = 0)
                
                specificity_output_dict = {
                    "predicted_ppm": predicted_ppm,
                    "true_sequence": feature_dict["S"][0].cpu().numpy().astype(np.int64),
                    "chain_labels": feature_dict["chain_labels"][0].cpu().numpy(),
                    "mask": feature_dict["mask"][0].cpu().numpy(),
                    "protein_mask": feature_dict["protein_mask"][0].cpu().numpy(),
                    "dna_mask": feature_dict["dna_mask"][0].cpu().numpy(),
                    "rna_mask": feature_dict["rna_mask"][0].cpu().numpy(),
                    "encoded_residues": encoded_residues,
                    "encoded_residues_dict": encoded_residue_dict,
                    "restype_to_int": restype_to_int
                }

                specificity_output_path = os.path.join(output_specificity, name + ".npz")
                np.savez(specificity_output_path, **specificity_output_dict)
            
            fasta_entires = []
            fasta_entires.append(
                '>{}, T={}, seed={}, num_res={}, batch_size={}, number_of_batches={}, model_path={}\n{}'.format(
                    name, 
                    args.temperature, 
                    seed, 
                    torch.sum(rec_mask).cpu().numpy(), 
                    args.batch_size, 
                    args.number_of_batches, 
                    checkpoint_path, 
                    seq_out_str
                )
            )
            for ix in range(S_stack.shape[0]):
                ix_suffix = ix
                if not args.zero_indexed:
                    ix_suffix += 1
                seq_rec_print = np.format_float_positional(rec_stack[ix].cpu().numpy(), unique=False, precision=4)
                loss_np = np.format_float_positional(np.exp(-loss_stack[ix].cpu().numpy()), unique=False, precision=4)
                loss_XY_np = np.format_float_positional(np.exp(-loss_XY_stack[ix].cpu().numpy()), unique=False, precision=4)

                #write new sequences into PDB with backbone coordinates
                seq_list = []
                for i, AA in enumerate(S_stack[ix].cpu().numpy()):
                    if feature_dict["rna_mask_for_token_conversion"][0, i] == 1:
                        seq_list.append(dna_char_to_rna_char.get(restype_INTtoSTR[AA], restype_INTtoSTR[AA]))
                    else:
                        seq_list.append(restype_INTtoSTR[AA])
                seq = "".join(seq_list)

                if args.output_pdbs:
                    if args.model_type == "na_mpnn":
                        seq_prody = np.array([restype_1to3[AA] for AA in list(seq)])
                        bfactor_prody = loss_per_residue_stack[ix].cpu().numpy()

                        for i, (chain_letter, res_idx) in enumerate(zip(chain_letters_list, R_idx_list)):
                            residue = backbone.select("chain {} and resnum {}".format(chain_letter, res_idx))
                            residue.setResnames(seq_prody[i])
                            residue.setBetas(np.exp(-bfactor_prody[i])*(bfactor_prody[i]>0.01).astype(np.float32))
                    else:
                        print("Choose --model_type flag from currently available models")
                        sys.exit()
                    
                    if other_atoms:
                        writePDB(output_backbones+name+'_'+str(ix_suffix)+".pdb"+ args.file_ending, backbone+other_atoms)
                    else:
                        writePDB(output_backbones+name+'_'+str(ix_suffix)+".pdb"+ args.file_ending, backbone)
                #-----
                
                #add fasta lines
                seq_np = np.array(list(seq))
                seq_out_str = []
                for mask in macromolecule_dict['mask_c']:
                    seq_out_str += list(seq_np[mask.cpu().numpy()])
                    seq_out_str += ['/']
                seq_out_str = "".join(seq_out_str)[:-1]
                fasta_entires.append(
                    '>{}, id={}, T={}, seed={}, overall_confidence={} seq_rec={}\n{}'.format(
                        name, 
                        ix_suffix,
                        args.temperature, 
                        seed, 
                        loss_np,
                        seq_rec_print, 
                        seq_out_str
                    )
                )
                #-----
            
            if args.output_sequences:
                with open(output_fasta, 'w') as f:
                    f.write("\n".join(fasta_entires))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--model_type", type=str, default="na_mpnn", help="Choose your model: na_mpnn")
    argparser.add_argument("--checkpoint_na_mpnn", type=str, default=None, help="Path to model weights.")

    argparser.add_argument("--out_folder", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--file_ending", type=str, default="", help="adding_string_to_the_end")
    argparser.add_argument("--pdb_path", type=str, default="", help="Path to the input PDB.")
    argparser.add_argument("--fixed_pos_by_pdb", type=str, default="", help="Path to json mapping of fixed residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}")
    argparser.add_argument("--zero_indexed", type=str, default=0, help="1 - to start output PDB numbering with 0")
    argparser.add_argument("--seed", type=int, default=0, help="Set seed for torch, numpy, and python random.")
    argparser.add_argument("--batch_size", type=int, default=None, help="Number of sequence to generate per one pass.")
    argparser.add_argument("--number_of_batches", type=int, default=1, help="Number of times to design sequence using a chosen batch size.")
    argparser.add_argument("--temperature", type=float, default=None, help="Temperature to sample sequences.")
    argparser.add_argument("--save_stats", type=int, default=0, help="Save output statistics")

    argparser.add_argument("--chains_to_design", type=str, default=None, help="Specify which chains to redesign, all others will be kept fixed.")
    argparser.add_argument("--omit_AA", type=str, default="X", help="Omit amino acids from design, e.g. XCG")
    argparser.add_argument("--fixed_residues", type=str, default="", help="Provide fixed residues, A12 A13 A14 B2 B25")
    argparser.add_argument("--redesigned_residues", type=str, default="", help="Provide to be redesigned residues, everything else will be fixed, A12 A13 A14 B2 B25")
    argparser.add_argument("--parse_these_chains_only", type=str, default="", help="Provide chains letters for parsing backbones, 'ABCF'")
    argparser.add_argument("--bias_AA", type=str, default="", help="Bias generation of amino acids, e.g. 'A:-1.024,P:2.34,C:-12.34'")
    argparser.add_argument("--pair_bias_AA", type=str, default="", help="Add pair bias for neighboring positions, e.g. 'KK:-10.0,KE:-10.0,EK:-10.0'")
    argparser.add_argument("--symmetry_residues", type=str, default="", help="Add list of lists for which residues need to be symmetric, e.g. 'A12,A13,A14|C2,C3|A5,B6'")
    argparser.add_argument("--symmetry_weights", type=str, default="", help="Add weights that match symmetry_residues, e.g. '1.01,1.0,1.0|-1.0,2.0|2.0,2.3'")

    argparser.add_argument("--na_shared_tokens", type=int, default=1, help="1 - use same tokens for RNA and DNA, 0 - separate tokens")
    argparser.add_argument("--parse_na_only", type=int, default=0, help="1 - to parse nucleic acid chains only, 0 - parse all chains")
    argparser.add_argument("--design_na_only", type=int, default=0, help="1 - to only design nucleic acid chains, 0 - do not limit to only nucleic acid chains")
    argparser.add_argument("--k_neighbors", type=int, default=None, help="If None, use number of nearest neighbors based on the model, otherwise override number of nearest neighbors with provided value")
    argparser.add_argument("--catch_failed_inferences", type=int, default=0, help="1 - catch exceptions raised during inference, saved failed inferences in separate folder, 0 - do not catch exceptions, do not save failed inferences")
    argparser.add_argument("--output_pdbs", type=int, default=1, help="1 - to output pdbs, 0 - do not output pdbs")
    argparser.add_argument("--output_sequences", type=int, default=1, help="1 - output sequences, 0 - do not output sequences")
    argparser.add_argument("--output_specificity", type=int, default=0, help="1 - output specificity scores, 0 - do not output specificity scores")
    argparser.add_argument("--load_residues_with_missing_atoms", type=int, default=0, help="1 - load residues with missing atoms, 0 - do not load residues with missing atoms")

    argparser.add_argument("--mode", type=str, default=None, help="Mode for NA-MPNN, choose from: design, specificity, None; if None, the user must provide --checkpoint_na_mpnn, --batch_size, and --temperature")

    args = argparser.parse_args()

    # Some defaults, depending on mode.
    if args.checkpoint_na_mpnn is None:
        if args.mode == "design":
            args.checkpoint_na_mpnn = "./models/design_model/s_19137.pt"
        elif args.mode == "specificity":
            args.checkpoint_na_mpnn = "./models/specificity_model/s_70114.pt"
        else:
            print("Choose mode from: design, specificity")
            sys.exit()
    if args.batch_size is None:
        if args.mode == "design":
            args.batch_size = 1
        elif args.mode == "specificity":
            args.batch_size = 30
        else:
            print("Choose mode from: design, specificity")
            sys.exit()
    if args.temperature is None:
        if args.mode == "design":
            args.temperature = 0.1
        elif args.mode == "specificity":
            args.temperature = 0.6
        else:
            print("Choose mode from: design, specificity")
            sys.exit()

    if not args.catch_failed_inferences:
        main(args)
    else:
        try:    
            main(args)
        except Exception as e:
            import os
            import json

            folder_for_outputs = args.out_folder
            base_folder = folder_for_outputs
            if base_folder[-1] != '/':
                base_folder = base_folder + '/'
            if not os.path.exists(base_folder + "failed_inferences"):
                os.makedirs(base_folder + "failed_inferences", exist_ok=True)
            output_failed_inferences = base_folder + '/failed_inferences/'

            if args.fixed_pos_by_pdb:
                with open(args.fixed_pos_by_pdb, 'r') as fh:
                    fixed_pos_by_pdb = json.load(fh)
            else:
                fixed_residues = [item for item in args.fixed_residues.split()]
                fixed_pos_by_pdb = {
                    args.pdb_path: fixed_residues
                }

            for pdb, fixed_residues in fixed_pos_by_pdb.items():
                name = pdb[pdb.rfind("/")+1:]
                if name[-4:] == ".pdb":
                    name = name[:-4]

                with open(output_failed_inferences + name + ".txt", mode = "w") as f:
                    f.write(str(e)) 
