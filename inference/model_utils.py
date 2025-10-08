from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import itertools
import sys

class ProteinMPNN(nn.Module):
    def __init__(self, 
                 num_letters=21, 
                 node_features=128, 
                 edge_features=128,
                 hidden_dim=128, 
                 num_encoder_layers=3, 
                 num_decoder_layers=3,
                 vocab=21, 
                 k_neighbors=48, 
                 augment_eps=0.0, 
                 dropout=0.0,
                 model_type="na_mpnn",
                 atom_dict=None,
                 restype_to_int=None,
                 polytype_to_int=None):
        super(ProteinMPNN, self).__init__()

        self.model_type = model_type
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.num_letters = num_letters

        if self.model_type == "na_mpnn":
            self.restype_to_int = restype_to_int
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
            self.features = ProteinFeaturesNA(node_features, 
                                              edge_features, 
                                              top_k=k_neighbors, 
                                              atom_dict=atom_dict, 
                                              polytype_to_int=polytype_to_int, 
                                              protein_augment_eps=augment_eps, 
                                              dna_augment_eps=augment_eps, 
                                              rna_augment_eps=augment_eps)
        else:
            print("Choose --model_type flag from currently available models")
            sys.exit()

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, feature_dict):
        #xyz_37 = feature_dict["xyz_37"] #[B,L,37,3] - xyz coordinates for all atoms if needed
        #xyz_37_m = feature_dict["xyz_37_m"] #[B,L,37] - mask for all coords
        #Y = feature_dict["Y"] #[B,L,num_context_atoms,3] - for ligandMPNN coords
        #Y_t = feature_dict["Y_t"] #[B,L,num_context_atoms] - element type
        #Y_m = feature_dict["Y_m"] #[B,L,num_context_atoms] - mask
        #X = feature_dict["X"] #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O 
        S_true = feature_dict["S"] #[B,L] - integer proitein sequence encoded using "restype_STRtoINT"
        #R_idx = feature_dict["R_idx"] #[B,L] - primary sequence residue index
        mask = feature_dict["mask"] #[B,L] - mask for missing regions - should be removed! all ones most of the time
        #chain_labels = feature_dict["chain_labels"] #[B,L] - integer labels for chain letters

        B, L = S_true.shape
        device = S_true.device

        if self.model_type == "na_mpnn":
            V, E, E_idx = self.features(feature_dict)
            h_V = self.W_v(V)
            h_E = self.W_e(E)

            mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            for layer in self.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        else:
            print("Choose --model_type flag from currently available models")
            sys.exit()

        return h_V, h_E, E_idx

    def sample(self, feature_dict):
        #xyz_37 = feature_dict["xyz_37"] #[B,L,37,3] - xyz coordinates for all atoms if needed
        #xyz_37_m = feature_dict["xyz_37_m"] #[B,L,37] - mask for all coords
        #Y = feature_dict["Y"] #[B,L,num_context_atoms,3] - for ligandMPNN coords
        #Y_t = feature_dict["Y_t"] #[B,L,num_context_atoms] - element type
        #Y_m = feature_dict["Y_m"] #[B,L,num_context_atoms] - mask
        #X = feature_dict["X"] #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O 
        B_decoder = feature_dict["batch_size"]
        S_true = feature_dict["S"] #[B,L] - integer proitein sequence encoded using "restype_STRtoINT"
        #R_idx = feature_dict["R_idx"] #[B,L] - primary sequence residue index
        mask = feature_dict["mask"] #[B,L] - mask for missing regions - should be removed! all ones most of the time
        chain_mask = feature_dict["chain_mask"] #[B,L] - mask for which residues need to be fixed; 0.0 - fixed; 1.0 - will be designed
        bias = feature_dict["bias"] #[B,L,self.num_letters] - amino acid bias per position
        pair_bias_flag = "pair_bias" in list(feature_dict)
        if pair_bias_flag:
            pair_bias = feature_dict["pair_bias"] #[B,L,self.num_letters,L,self.num_letters] - pair amino acid bias per position
        #chain_labels = feature_dict["chain_labels"] #[B,L] - integer labels for chain letters
        randn = feature_dict["randn"] #[B,L] - random numbers for decoding order; only the first entry is used since decoding within a batch needs to match for symmetry
        temperature = feature_dict["temperature"] #float - sampling temperature; prob = softmax(logits/temperature)
        symmetry_list_of_lists = feature_dict["symmetry_residues"] #[[0, 1, 14], [10,11,14,15], [self.num_letters - 1, self.num_letters]] #indices to select X over length - L
        symmetry_weights_list_of_lists = feature_dict["symmetry_weights"] #[[1.0, 1.0, 1.0], [-2.0,1.1,0.2,1.1], [2.3, 1.1]]

        B, L = S_true.shape
        device = S_true.device

        h_V, h_E, E_idx = self.encode(feature_dict)

        chain_mask = mask*chain_mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        if len(symmetry_list_of_lists[0])==0 and len(symmetry_list_of_lists)==1:
            E_idx = E_idx.repeat(B_decoder, 1, 1)
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=L).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(L,L, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([B, L, 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)

            #repeat for decoding
            S_true = S_true.repeat(B_decoder, 1)
            h_V = h_V.repeat(B_decoder, 1, 1)
            h_E = h_E.repeat(B_decoder, 1, 1, 1)
            chain_mask = chain_mask.repeat(B_decoder, 1)
            mask = mask.repeat(B_decoder, 1)
            bias = bias.repeat(B_decoder, 1, 1)
            if pair_bias_flag:
                pair_bias = pair_bias.repeat(B_decoder, 1, 1, 1, 1)
            #-----
            
            if self.model_type == "na_mpnn":
                all_probs = torch.zeros((B_decoder, L, self.num_letters), device=device, dtype=torch.float32)
            else:
                print("Choose --model_type flag from currently available models")
                sys.exit()
            all_log_probs = torch.zeros((B_decoder, L, self.num_letters), device=device, dtype=torch.float32)
            h_S = torch.zeros_like(h_V, device=device)
            S = (self.num_letters - 1)*torch.ones((B_decoder, L), dtype=torch.int64, device=device)
            h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]

            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder

            for t_ in range(L):
                t = decoding_order[:,t_] #[B]
                chain_mask_t = torch.gather(chain_mask, 1, t[:,None])[:,0] #[B]
                mask_t = torch.gather(mask, 1, t[:,None])[:,0] #[B]
                bias_t = torch.gather(bias, 1, t[:,None,None].repeat(1,1,self.num_letters))[:,0,:] #[B,self.num_letters]
                if pair_bias_flag:
                    pair_bias_t = torch.gather(pair_bias, 1, t[:,None,None,None,None].repeat(1,1,self.num_letters,L,self.num_letters))[:,0,:,:,:] #[B, self.num_letters, L, self.num_letters]
                    pair_bias_t = torch.gather(pair_bias_t, -1, S[:,None,:,None].repeat(1,self.num_letters,1,1))[:,:,:,0] #[B, self.num_letters, L]
                    pair_bias_t = pair_bias_t.sum(-1) #[B, self.num_letters]


                E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))

                mask_bw_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1]))

                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                    h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
            
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
                logits = self.W_out(h_V_t) #[B,self.num_letters]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1) #[B,self.num_letters]

                
                if pair_bias_flag:
                    probs = torch.nn.functional.softmax((logits+bias_t+pair_bias_t) / temperature, dim=-1) #[B,self.num_letters]
                else:
                    probs = torch.nn.functional.softmax((logits+bias_t) / temperature, dim=-1) #[B,self.num_letters]

                if self.model_type == "na_mpnn":
                    probs[:, self.restype_to_int["UNK"]] = 0
                    probs[:, self.restype_to_int["DX"]] = 0
                    probs[:, self.restype_to_int["RX"]] = 0
                    probs[:, self.restype_to_int["MAS"]] = 0
                    probs[:, self.restype_to_int["PAD"]] = 0

                    probs_sample = probs/torch.sum(probs, dim=-1, keepdim=True)
                else:
                    print("Choose --model_type flag from currently available models")
                    sys.exit()
                S_t = torch.multinomial(probs_sample, 1)[:,0] #[B]

                all_probs.scatter_(1, t[:,None,None].repeat(1,1,self.num_letters - 1), (chain_mask_t[:,None,None]*probs_sample[:,None,:]).float())
                all_log_probs.scatter_(1, t[:,None,None].repeat(1,1,self.num_letters), (chain_mask_t[:,None,None]*log_probs[:,None,:]).float())
                S_true_t = torch.gather(S_true, 1, t[:,None])[:,0]
                S_t = (S_t*chain_mask_t+S_true_t*(1.0-chain_mask_t)).long()
                h_S.scatter_(1, t[:,None,None].repeat(1,1,h_S.shape[-1]), self.W_s(S_t)[:,None,:])
                S.scatter_(1, t[:,None], S_t[:,None])
                
            output_dict = {"S": S, "sampling_probs": all_probs, "log_probs": all_log_probs, "decoding_order": decoding_order}
        else:
            #weights for symmetric design
            symmetry_weights = torch.ones([L], device=device, dtype=torch.float32)
            for i1, item_list in enumerate(symmetry_list_of_lists):
                for i2, item in enumerate(item_list):
                    symmetry_weights[item] = symmetry_weights_list_of_lists[i1][i2]

            new_decoding_order = []
            for t_dec in list(decoding_order[0,].cpu().data.numpy()):
                if t_dec not in list(itertools.chain(*new_decoding_order)):
                    list_a = [item for item in symmetry_list_of_lists if t_dec in item]
                    if list_a:
                        new_decoding_order.append(list_a[0])
                    else:
                        new_decoding_order.append([t_dec])

            decoding_order = torch.tensor(list(itertools.chain(*new_decoding_order)), device=device)[None,].repeat(B,1)

            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=L).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(L,L, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([B, L, 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)

            #repeat for decoding
            S_true = S_true.repeat(B_decoder, 1)
            h_V = h_V.repeat(B_decoder, 1, 1)
            h_E = h_E.repeat(B_decoder, 1, 1, 1)
            E_idx = E_idx.repeat(B_decoder, 1, 1)
            mask_fw = mask_fw.repeat(B_decoder, 1, 1, 1)
            mask_bw = mask_bw.repeat(B_decoder, 1, 1, 1)
            chain_mask = chain_mask.repeat(B_decoder, 1)
            mask = mask.repeat(B_decoder, 1)
            bias = bias.repeat(B_decoder, 1, 1)
            if pair_bias_flag:
                pair_bias = pair_bias.repeat(B_decoder, 1, 1, 1, 1)
            #-----

            if self.model_type == "na_mpnn":
                all_probs = torch.zeros((B_decoder, L, self.num_letters), device=device, dtype=torch.float32)
            else:
                print("Choose --model_type flag from currently available models")
                sys.exit()
            all_log_probs = torch.zeros((B_decoder, L, self.num_letters), device=device, dtype=torch.float32)
            h_S = torch.zeros_like(h_V, device=device)
            S = (self.num_letters - 1)*torch.ones((B_decoder, L), dtype=torch.int64, device=device)
            h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]

            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder

            for t_list in new_decoding_order:
                total_logits=0.0
                logits_list = []
                for t in t_list:
                    chain_mask_t = chain_mask[:,t] #[B]
                    mask_t = mask[:,t] #[B]
                    bias_t = bias[:,t] #[B, self.num_letters]
                    if pair_bias_flag:
                        pair_bias_t = pair_bias[:,t] #[B, self.num_letters, L, self.num_letters]
                        pair_bias_t = torch.gather(pair_bias_t, -1, S[:,None,:,None].repeat(1,self.num_letters,1,1))[:,:,:,0] #[B, self.num_letters, L]
                        pair_bias_t = pair_bias_t.sum(-1) #[B, self.num_letters]

                    E_idx_t = E_idx[:,t:t+1]
                    h_E_t = h_E[:,t:t+1]
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = h_EXV_encoder_fw[:,t:t+1]
                    for l, layer in enumerate(self.decoder_layers):
                        h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                        h_V_t = h_V_stack[l][:,t:t+1]
                        h_ESV_t = mask_bw[:,t:t+1]* h_ESV_decoder_t + h_EXV_encoder_t
                        h_V_stack[l+1][:,t:t+1,:] = layer(h_V_t, h_ESV_t, mask_V=mask_t[:,None])
                
                    h_V_t = h_V_stack[-1][:,t] 
                    logits = self.W_out(h_V_t) #[B,self.num_letters]
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1) #[B,self.num_letters]
                    all_log_probs[:,t] = (chain_mask_t[:,None]*log_probs).float() #[B,self.num_letters]
                    total_logits += symmetry_weights[t]*logits

                if pair_bias_flag:
                    probs = torch.nn.functional.softmax((total_logits+bias_t+pair_bias_t) / temperature, dim=-1) #[B,self.num_letters]
                else:
                    probs = torch.nn.functional.softmax((total_logits+bias_t) / temperature, dim=-1) #[B,self.num_letters]

                if self.model_type == "na_mpnn":
                    probs[:, self.restype_to_int["UNK"]] = 0
                    probs[:, self.restype_to_int["DX"]] = 0
                    probs[:, self.restype_to_int["RX"]] = 0
                    probs[:, self.restype_to_int["MAS"]] = 0
                    probs[:, self.restype_to_int["PAD"]] = 0

                    probs_sample = probs/torch.sum(probs, dim=-1, keepdim=True)
                else:
                    print("Choose --model_type flag from currently available models")
                    sys.exit()

                S_t = torch.multinomial(probs_sample, 1)[:,0] #[B]
                for t in t_list:
                    chain_mask_t = chain_mask[:,t] #[B]
                    all_probs[:,t] = (chain_mask_t[:,None]*probs_sample).float() #[B,self.num_letters - 1]
                    S_true_t = S_true[:,t] #[B]
                    S_t = (S_t*chain_mask_t+S_true_t*(1.0-chain_mask_t)).long()
                    h_S[:,t] = self.W_s(S_t)
                    S[:,t] = S_t
                
            output_dict = {"S": S, "sampling_probs": all_probs, "log_probs": all_log_probs, "decoding_order": decoding_order.repeat(B_decoder,1)}
        return output_dict

    def unconditional_probs(self, feature_dict):
        #xyz_37 = feature_dict["xyz_37"] #[B,L,37,3] - xyz coordinates for all atoms if needed
        #xyz_37_m = feature_dict["xyz_37_m"] #[B,L,37] - mask for all coords
        #Y = feature_dict["Y"] #[B,L,num_context_atoms,3] - for ligandMPNN coords
        #Y_t = feature_dict["Y_t"] #[B,L,num_context_atoms] - element type
        #Y_m = feature_dict["Y_m"] #[B,L,num_context_atoms] - mask
        X = feature_dict["X"] #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O 
        B_decoder = feature_dict["batch_size"]
        #R_idx = feature_dict["R_idx"] #[B,L] - primary sequence residue index
        mask = feature_dict["mask"] #[B,L] - mask for missing regions - should be removed! all ones most of the time
        #chain_labels = feature_dict["chain_labels"] #[B,L] - integer labels for chain letters
        device=X.device

        h_V, h_E, E_idx = self.encode(feature_dict)
        order_mask_backward = torch.zeros([X.shape[0], X.shape[1], X.shape[1]], device=device)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_fw = mask_1D * (1. - mask_attend)

        h_V = h_V.repeat(B_decoder, 1, 1)
        h_E = h_E.repeat(B_decoder, 1, 1, 1)
        E_idx = E_idx.repeat(B_decoder, 1, 1)
        mask_fw = mask_fw.repeat(B_decoder, 1, 1, 1)
        mask = mask.repeat(B_decoder, 1)

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.W_out(h_V)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        output_dict = {"log_probs": log_probs}
        return output_dict

    def score(self, feature_dict):
        #check if score matches - sample log probs

        #xyz_37 = feature_dict["xyz_37"] #[B,L,37,3] - xyz coordinates for all atoms if needed
        #xyz_37_m = feature_dict["xyz_37_m"] #[B,L,37] - mask for all coords
        #Y = feature_dict["Y"] #[B,L,num_context_atoms,3] - for ligandMPNN coords
        #Y_t = feature_dict["Y_t"] #[B,L,num_context_atoms] - element type
        #Y_m = feature_dict["Y_m"] #[B,L,num_context_atoms] - mask
        #X = feature_dict["X"] #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O 
        B_decoder = feature_dict["batch_size"]
        S_true = feature_dict["S"] #[B,L] - integer proitein sequence encoded using "restype_STRtoINT"
        #R_idx = feature_dict["R_idx"] #[B,L] - primary sequence residue index
        mask = feature_dict["mask"] #[B,L] - mask for missing regions - should be removed! all ones most of the time
        chain_mask = feature_dict["chain_mask"] #[B,L] - mask for which residues need to be fixed; 0.0 - fixed; 1.0 - will be designed
        #chain_labels = feature_dict["chain_labels"] #[B,L] - integer labels for chain letters
        randn = feature_dict["randn"] #[B,L] - random numbers for decoding order; only the first entry is used since decoding within a batch needs to match for symmetry

        B, L = S_true.shape
        device = S_true.device

        h_V, h_E, E_idx = self.encode(feature_dict)

        chain_mask = mask*chain_mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]

        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=L).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(L,L, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([B, L, 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        #repeat for decoding
        S_true = S_true.repeat(B_decoder, 1)
        h_V = h_V.repeat(B_decoder, 1, 1)
        h_E = h_E.repeat(B_decoder, 1, 1, 1)
        E_idx = E_idx.repeat(B_decoder, 1, 1)
        chain_mask = chain_mask.repeat(B_decoder, 1)
        mask = mask.repeat(B_decoder, 1)

        h_S = self.W_s(S_true)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        output_dict = {"S": S_true, "log_probs": log_probs, "decoding_order": decoding_order[0]}
        return output_dict

class ProteinFeaturesNA(nn.Module):
    def __init__(self, 
                 edge_features, 
                 node_features, 
                 num_positional_embeddings=16,
                 num_rbf=16, 
                 top_k=30,
                 atom_dict=None,
                 polytype_to_int=None,
                 protein_augment_eps=0., 
                 dna_augment_eps=0.,
                 rna_augment_eps=0.,
                 na_ref_atom="C1'",
                 include_pred_na_N=1):
        """ Extract protein features """
        super(ProteinFeaturesNA, self).__init__()

        if atom_dict is None:
            raise Exception("atom_dict is necessary for featurization!")
    
        if polytype_to_int is None:
            raise Exception("polytype_to_int is necessary for featurization!")

        self.N_idx = atom_dict["N"]
        self.CA_idx = atom_dict["CA"]
        self.C_idx = atom_dict["C"]

        self.O4prime_idx = atom_dict["O4'"]
        self.C1prime_idx = atom_dict["C1'"]
        self.C2prime_idx = atom_dict["C2'"]

        self.na_ref_atom_idx = atom_dict[na_ref_atom]

        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k

        self.protein_augment_eps = protein_augment_eps
        self.dna_augment_eps = dna_augment_eps
        self.rna_augment_eps = rna_augment_eps

        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)

        self.num_polytypes = len(polytype_to_int)
        self.node_in = len(polytype_to_int)

        self.node_embedding = nn.Linear(self.node_in, node_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)

        total_atoms = len(atom_dict) + 1

        self.include_pred_na_N = include_pred_na_N
        if self.include_pred_na_N:
            total_atoms = total_atoms + 1

        self.edge_in = num_positional_embeddings + num_rbf*total_atoms*total_atoms
        self.edge_embedding = nn.Linear(self.edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)


    def _dist(self, X, mask, eps = 1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_all_rbf(self, X, E_idx, X_m):
        # [B,L,16,3] => [B,L,16*3] => [B,L,K,16*3] => [B,L,K,16,3] => [B,L,K,16,16]
        X_flat = X.reshape((X.shape[0], X.shape[1], -1))
        X_flat_g = gather_nodes(X_flat, E_idx) #[B,L,K,16*3]
        X_g = X_flat_g.reshape(list(X_flat_g.shape)[:-1] + list(X.shape[-2:])) #[B,L,K,16,3]
        D = torch.sqrt(torch.sum((X[:,:,None,:,None,:] - X_g[:,:,:,None,:,:])**2,-1) + 1e-6)
        RBF_all = self._rbf(D) #[B, L, K, 16, 16, H]
        X_m_gathered = gather_nodes(X_m, E_idx) #[B,L,K,16]
        RBF_all =  RBF_all*X_m[:,:,None,:,None,None]*X_m_gathered[:,:,:,None,:,None] #[B,L,K,16,16,H]
        RBF_all = RBF_all.view([X.shape[0], X.shape[1], E_idx.shape[2],-1])  #[B,L,K,16*16*H]
        return RBF_all
    
    def get_Cb(self, N, Ca, C, w_a, w_b, w_c):
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = w_a * a + w_b * b + w_c * c + Ca #shift from CA
        return Cb

    def forward(self, feature_dict):
        X = feature_dict["X"] 
        mask = feature_dict["mask"]
        R_idx = feature_dict["R_idx"]
        chain_labels = feature_dict["chain_labels"]
        X_m = feature_dict["X_m"]
        protein_mask = feature_dict["protein_mask"]
        dna_mask = feature_dict["dna_mask"]
        rna_mask = feature_dict["rna_mask"]
        R_polymer_type = feature_dict["R_polymer_type"]

        if self.training and (self.protein_augment_eps > 0 or \
                              self.dna_augment_eps > 0 or \
                              self.rna_augment_eps > 0):            
            augment_eps = protein_mask * self.protein_augment_eps + \
                dna_mask * self.dna_augment_eps + \
                rna_mask * self.rna_augment_eps
            
            X = X + X_m[:,:,:,None] * augment_eps[:,:,None,None] * torch.randn_like(X)
        
        Ca = X[:,:,self.CA_idx,:]
        N = X[:,:,self.N_idx,:]
        C = X[:,:,self.C_idx,:]

        Cb = self.get_Cb(N, Ca, C, w_a = -0.58273431, w_b = 0.56802827, w_c = -0.54067466)

        na_ref_atom = X[:,:,self.na_ref_atom_idx,:]
        if self.include_pred_na_N:
            O4prime = X[:,:,self.O4prime_idx,:]
            C1prime = X[:,:,self.C1prime_idx,:]
            C2prime = X[:,:,self.C2prime_idx,:]

            N_na = self.get_Cb(O4prime, C1prime, C2prime, w_a = -0.56967352, w_b = 0.51055973, w_c = -0.53122153)

            augmented_X = (X, Cb[:,:,None,:], N_na[:,:,None,:])
            augmented_X_m = (X_m, protein_mask[:,:,None], (rna_mask + dna_mask)[:,:,None])
        else:
            augmented_X = (X, Cb[:,:,None,:])
            augmented_X_m = (X_m, protein_mask[:,:,None])

        augmented_X = torch.cat(augmented_X, -2)
        augmented_X_m = torch.cat(augmented_X_m, -1)

        # Ca + P because these vectors are disjoint. This sum represents the
        # center coordinates for all (protein or dna) residues.
        D_neighbors, E_idx = self._dist(Ca + na_ref_atom, mask)

        RBF_all = self._get_all_rbf(augmented_X, E_idx, augmented_X_m)

        offset = R_idx[:,:,None]-R_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        R_polymer_type_one_hot = torch.nn.functional.one_hot(R_polymer_type, num_classes = self.num_polytypes).float()
        V = R_polymer_type_one_hot

        V = self.node_embedding(V)
        V = self.norm_nodes(V)

        return V, E, E_idx

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E

# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn