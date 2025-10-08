from openbabel import openbabel
openbabel.obErrorLog.SetOutputLevel(0)
openbabel.cvar.obErrorLog.StopLogging()
import numpy as np
import pandas as pd
import sys
import torch
import time
import json
import os

import cifutils
import pdbutils
from na_data_utils import PDBDataset, make_batch_iter 
from na_model_utils import featurize, loss_smoothed, loss_nll, compute_canonical_base_pair_accuracy, get_std_opt, ProteinMPNN
from na_metric_manager import generate_metric_manager

JSON = sys.argv[1]
params = json.load(open(JSON))

scaler = torch.cuda.amp.GradScaler()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

if params["BASE_FOLDER"][-1] != '/':
    params["BASE_FOLDER"] += '/'
if not os.path.exists(params["BASE_FOLDER"]):
    os.makedirs(params["BASE_FOLDER"])

logfile = params["BASE_FOLDER"] + 'log.txt'
if not params["PREV_CHECKPOINT"]:
    with open(logfile, 'w') as f:
        f.write('Epoch\tTrain\tValidation\n')

if params["ATOMS_TO_LOAD"] == "backbone":
    atom_list_to_save = ['N', 'CA', 'C', 'O', #protein atoms
                         'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'" #nucleic acid atoms
                        ]
elif params["ATOMS_TO_LOAD"] == "all":
    atom_list_to_save = ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT', #protein atoms
                         'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N9', 'C8', 'C7', 'N7', 'C6', 'N6', 'O6', 'C5', 'C4', 'N4', 'O4', 'N3', 'C2', 'N2', 'O2', 'N1' #nucleic acid atoms
                        ]

cif_parser = cifutils.CIFParser(skip_res=params["EXCLUDE_RES"], randomize_nmr_model=params["RANDOMIZE_NMR_MODEL"])
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
                         na_ref_atom=params["NA_REF_ATOM"],
                         parse_ppms=params["PARSE_PPMS"],
                         min_overlap_length=params["MIN_OVERLAP_LENGTH"],
                         drop_protein_probability=params["DROP_PROTEIN_PROBABILITY"],
                         na_only_as_uniform_ppm=params["NA_ONLY_AS_UNIFORM_PPM"],
                         protein_interface_residue_mutation_probability=params["PROTEIN_INTERFACE_RESIDUE_MUTATION_PROBABILITY"],
                         mutate_base_pair_together=params["MUTATE_BASE_PAIR_TOGETHER"],
                         mutate_entire_side_chain_interface_probability=params["MUTATE_ENTIRE_SIDE_CHAIN_INTERFACE_PROBABILITY"],
                         na_non_interface_as_uniform_ppm=params["NA_NON_INTERFACE_AS_UNIFORM_PPM"]
                         )

model = ProteinMPNN(node_features=params["HIDDEN_DIM"],
                    edge_features=params["HIDDEN_DIM"],
                    hidden_dim=params["HIDDEN_DIM"],
                    num_encoder_layers=params["NUM_ENCODER_LAYERS"],
                    num_decoder_layers=params["NUM_DECODER_LAYERS"],
                    k_neighbors=params["NUM_NEIGHBORS"],
                    dropout=params["DROPOUT"],
                    atom_dict=pdb_dataset.atom_dict,
                    restype_to_int=pdb_dataset.restype_to_int,
                    polytype_to_int=pdb_dataset.polytype_to_int,
                    protein_augment_eps=params["PROTEIN_BACKBONE_NOISE"],
                    dna_augment_eps=params["DNA_BACKBONE_NOISE"],
                    rna_augment_eps=params["RNA_BACKBONE_NOISE"],
                    decode_protein_first=params["DECODE_PROTEIN_FIRST"],
                    na_ref_atom=params["NA_REF_ATOM"],
                    include_pred_na_N=params["INCLUDE_PRED_NA_N"],
                    device=device,
                    vocab=params["VOCAB_SIZE"],
                    num_letters=params["NUM_LETTERS"])
model.to(device)

if params["PREV_CHECKPOINT"]:
    try:
        checkpoint = torch.load(params["PREV_CHECKPOINT"])
        total_step = checkpoint['step'] #write total_step from the checkpoint
        save_step = checkpoint['save_step']
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Starting from step {total_step}")
    except:
        print("LOADING FROM BAD PATH CHECKPOINT")
        total_step = 0
        epoch = 0
        save_step = 0
        params["PREV_CHECKPOINT"] = []
else:
    total_step = 0
    epoch = 0
    save_step = 0


optimizer = get_std_opt(model.parameters(), params["HIDDEN_DIM"], total_step)

if params["PREV_CHECKPOINT"]:
    optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

df_valid = pd.read_csv(params["DF_PATH_VALID"])
df_train = pd.read_csv(params["DF_PATH_TRAIN"])

# Convert the dates to datetime.
df_valid["date"] = pd.to_datetime(df_valid["date"], format = "%Y-%m-%d")
df_train["date"] = pd.to_datetime(df_train["date"], format = "%Y-%m-%d")

# Convert the date cutoff to datetime.
date_cutoff = pd.to_datetime(params["DATE_CUTOFF"], format = "%Y-%m-%d")

metric_manager = generate_metric_manager(pdb_dataset.restype_to_int, metrics_to_compute = params["METRICS_TO_COMPUTE"])

tokens_with_no_loss = torch.tensor([pdb_dataset.restype_to_int["UNK"], 
                                    pdb_dataset.restype_to_int["DX"], 
                                    pdb_dataset.restype_to_int["RX"], 
                                    pdb_dataset.restype_to_int["MAS"], 
                                    pdb_dataset.restype_to_int["PAD"]], 
                                   device=device)

# Masks used for loss function.
protein_restype_mask = torch.zeros(params["NUM_LETTERS"], device=device)
protein_restype_mask[pdb_dataset.protein_restype_ints] = 1

dna_restype_mask = torch.zeros(params["NUM_LETTERS"], device=device)
dna_restype_mask[pdb_dataset.dna_restype_ints] = 1
    
rna_restype_mask = torch.zeros(params["NUM_LETTERS"], device=device)
rna_restype_mask[pdb_dataset.rna_restype_ints] = 1

polymer_restype_masks = {"protein": protein_restype_mask,
                         "dna": dna_restype_mask,
                         "rna": rna_restype_mask}

polymer_restype_nums = {"protein": len(pdb_dataset.protein_restype_ints),
                        "dna": len(pdb_dataset.dna_restype_ints),
                        "rna": len(pdb_dataset.rna_restype_ints)}
          
for e in range(100000):
    metric_manager.zero_metrics()

    batch_iter_valid = make_batch_iter(df = df_valid, 
                                       batch_tokens = params["BATCH_TOKENS"], 
                                       length_cutoff = params["MIN_PROTEIN_LENGTH_CUTOFF"],
                                       date_cutoff = date_cutoff, 
                                       crop_large_structures = params["CROP_LARGE_STRUCTURES"],
                                       max_number_of_pdbs = params["MAX_NUMBER_OF_PDBS_VALID"])
    
    batch_iter_train = make_batch_iter(df = df_train, 
                                       batch_tokens = params["BATCH_TOKENS"], 
                                       length_cutoff = params["MIN_PROTEIN_LENGTH_CUTOFF"],
                                       date_cutoff = date_cutoff, 
                                       crop_large_structures = params["CROP_LARGE_STRUCTURES"],
                                       max_number_of_pdbs = params["MAX_NUMBER_OF_PDBS_TRAIN"])

    valid_sampler = torch.utils.data.sampler.BatchSampler(
        batch_iter_valid,
        batch_size=1,
        drop_last=False)

    train_sampler = torch.utils.data.sampler.BatchSampler(
        batch_iter_train,
        batch_size=1,
        drop_last=False)

    valid_loader = torch.utils.data.DataLoader(
        pdb_dataset,
        sampler=valid_sampler,
        num_workers=params["NUM_WORKERS"],
        pin_memory=False)

    train_loader = torch.utils.data.DataLoader(
        pdb_dataset,
        sampler=train_sampler,
        num_workers=params["NUM_WORKERS"],
        pin_memory=False)

    model.train()
    e = epoch + e
    t0 = time.time()
    for ix, batch in enumerate(train_loader):
        optimizer.zero_grad()
        feature_dict = featurize(batch, pdb_dataset.polytype_to_int, pdb_dataset.restype_to_int, pdb_dataset.atom_dict, device)
        if type(feature_dict) == str:
            continue
        S = feature_dict["S"]
        mask = feature_dict["mask"]
        S_mask = 1 - (torch.any(S[:,:,None] == tokens_with_no_loss[None,None,:], dim = -1)).long()
        mask_for_loss = mask * S_mask
        feature_dict["mask_for_loss"] = mask_for_loss

        polymer_masks = {"protein": feature_dict["protein_mask"], "dna": feature_dict["dna_mask"], "rna": feature_dict["rna_mask"]}
        if params["METRICS_TO_COMPUTE"] == "all":
            interface_masks = {"interface": feature_dict["interface_mask"],
                               "nonInterface": 1 - feature_dict["interface_mask"]}
        else:
            interface_masks = {}

        if params["MIXED_PRECISION"]:
            with torch.cuda.amp.autocast():
                log_probs, probs = model(feature_dict)

                _, loss_av_smoothed = loss_smoothed(S, 
                                                    log_probs, 
                                                    mask_for_loss,
                                                    polymer_masks=polymer_masks, 
                                                    polymer_restype_masks=polymer_restype_masks, 
                                                    polymer_restype_nums=polymer_restype_nums, 
                                                    weight=params["LABEL_SMOOTHING"], 
                                                    tokens=params["LOSS_TOKENS"], 
                                                    num_letters=params["NUM_LETTERS"],
                                                    ppm_mask=feature_dict["ppm_mask"],
                                                    aligned_ppm=feature_dict["aligned_ppm"])

            scaler.scale(loss_av_smoothed).backward()

            if params["GRADIENT_NORM"] > 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), params["GRADIENT_NORM"])

            scaler.step(optimizer)
            scaler.update()

        loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)        
        canonical_base_pair_accuracy = \
            compute_canonical_base_pair_accuracy(log_probs, 
                                                 feature_dict["canonical_base_pair_mask"], 
                                                 feature_dict["canonical_base_pair_index"], 
                                                 pdb_dataset)
        
        S_pred = torch.argmax(log_probs, -1)

        loss_for_metric, _ = loss_smoothed(S, 
                                           log_probs, 
                                           mask_for_loss,
                                           polymer_masks=polymer_masks, 
                                           polymer_restype_masks=polymer_restype_masks, 
                                           polymer_restype_nums=polymer_restype_nums, 
                                           weight=params["LABEL_SMOOTHING"], 
                                           tokens=params["LOSS_TOKENS"], 
                                           num_letters=params["NUM_LETTERS"],
                                           ppm_mask=feature_dict["ppm_mask"],
                                           aligned_ppm=feature_dict["aligned_ppm"])

        metric_manager.accumulate(loss_for_metric, 
                                  true_false, 
                                  canonical_base_pair_accuracy, 
                                  feature_dict["canonical_base_pair_mask"], 
                                  S, 
                                  S_pred, 
                                  "train", 
                                  mask_for_loss, 
                                  polymer_masks, 
                                  interface_masks)

        total_step += 1

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for ix, batch in enumerate(valid_loader):
            feature_dict = featurize(batch, pdb_dataset.polytype_to_int, pdb_dataset.restype_to_int, pdb_dataset.atom_dict, device)
            if type(feature_dict) == str:
                continue
            S = feature_dict["S"]
            mask = feature_dict["mask"]
            S_mask = 1 - (torch.any(S[:,:,None] == tokens_with_no_loss[None,None,:], dim = -1)).long()
            mask_for_loss = mask * S_mask
            feature_dict["mask_for_loss"] = mask_for_loss

            polymer_masks = {"protein": feature_dict["protein_mask"], "dna": feature_dict["dna_mask"], "rna": feature_dict["rna_mask"]}
            if params["METRICS_TO_COMPUTE"] == "all":
                interface_masks = {"interface": feature_dict["interface_mask"],
                                   "nonInterface": 1 - feature_dict["interface_mask"]}
            else:
                interface_masks = {}

            log_probs, probs = model(feature_dict)
            
            loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            canonical_base_pair_accuracy = \
                compute_canonical_base_pair_accuracy(log_probs, 
                                                     feature_dict["canonical_base_pair_mask"], 
                                                     feature_dict["canonical_base_pair_index"],
                                                     pdb_dataset)
            S_pred = torch.argmax(log_probs, -1)

            loss_for_metric, _ = loss_smoothed(S, 
                                               log_probs, 
                                               mask_for_loss,
                                               polymer_masks=polymer_masks, 
                                               polymer_restype_masks=polymer_restype_masks, 
                                               polymer_restype_nums=polymer_restype_nums, 
                                               weight=params["LABEL_SMOOTHING"], 
                                               tokens=params["LOSS_TOKENS"], 
                                               num_letters=params["NUM_LETTERS"],
                                               ppm_mask=feature_dict["ppm_mask"],
                                               aligned_ppm=feature_dict["aligned_ppm"])

            metric_manager.accumulate(loss_for_metric, 
                                      true_false, 
                                      canonical_base_pair_accuracy, 
                                      feature_dict["canonical_base_pair_mask"], 
                                      S, 
                                      S_pred, 
                                      "valid", 
                                      mask_for_loss, 
                                      polymer_masks, 
                                      interface_masks)

    t2 = time.time()
    metric_manager.compute_metrics()

    train_dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=3)
    valid_dt = np.format_float_positional(np.float32(t2-t1), unique=False, precision=3)

    output_string = metric_manager.create_print_string(e, total_step, train_dt, 
                                                       valid_dt)

    with open(logfile, 'a') as f:
        f.write(output_string + "\n")
    print(output_string)
    torch.save({'epoch': e+1,
                'step': total_step,
                'save_step': save_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                }, params["BASE_FOLDER"]+'last.pt')
    
    if total_step > save_step + params["SAVE_EVERY_N_STEPS"]:
        save_step += params["SAVE_EVERY_N_STEPS"]
        torch.save({'epoch': e+1,
                    'step': total_step,
                    'save_step': save_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, params["BASE_FOLDER"]+f's_{total_step}.pt')
    if total_step > params["TOTAL_STEPS"]:
        break
