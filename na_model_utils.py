from __future__ import print_function
import numpy as np
import torch
import torch.utils
import torch.utils.checkpoint
import torch.nn as nn

def featurize(batch, polytype_to_int, restype_to_int, atom_dict, device):
    batch = [b for b in batch if type(b[0])!=list]
    B = len(batch)
    if B > 0:
        L_stack = torch.stack([b[1] for b in batch])
        L_max = torch.max(L_stack)
        X = torch.zeros([B, L_max, len(atom_dict), 3], dtype=torch.float32)
        X_m = torch.zeros([B, L_max, len(atom_dict)], dtype=torch.int32)

        mask = torch.zeros([B, L_max], dtype=torch.int32)
        S = restype_to_int["PAD"] * torch.ones([B, L_max], dtype=torch.int64)
        R_idx = -100*torch.ones([B, L_max], dtype=torch.int32)
        chain_labels = -1*torch.ones([B, L_max], dtype=torch.int64)

        protein_mask = torch.zeros([B, L_max], dtype=torch.int32)
        dna_mask = torch.zeros([B, L_max], dtype=torch.int32)
        rna_mask = torch.zeros([B, L_max], dtype=torch.int32)

        R_polymer_type = polytype_to_int["PAD"] * torch.ones([B, L_max], dtype=torch.int64)

        interface_mask = torch.zeros([B, L_max], dtype=torch.int32)
        base_pair_mask = torch.zeros([B, L_max], dtype=torch.int32)
        base_pair_index = torch.zeros([B, L_max], dtype=torch.int64)
        canonical_base_pair_mask = torch.zeros([B, L_max], dtype=torch.int32)
        canonical_base_pair_index = torch.zeros([B, L_max], dtype=torch.int64)

        aligned_ppm = torch.zeros([B, L_max, len(restype_to_int)], dtype=torch.float64)
        ppm_mask = torch.zeros([B, L_max], dtype=torch.int32)

        structure_paths = []
        assembly_ids = []

        for i, b in enumerate(batch):
            out_dict = b[0]
            X[i,:L_stack[i]] = out_dict["X"][None,]
            X_m[i,:L_stack[i]] = out_dict["X_m"][None,]
            mask[i,:L_stack[i]] = torch.ones_like(out_dict["S"][None,], dtype=torch.int32)
            S[i,:L_stack[i]] = out_dict["S"][None,]
            R_idx[i,:L_stack[i]] = out_dict["R_idx"][None,]
            chain_labels[i,:L_stack[i]] = out_dict["chain_labels"][None,]

            protein_mask[i,:L_stack[i]] = out_dict["protein_mask"][None,]
            dna_mask[i,:L_stack[i]] = out_dict["dna_mask"][None,]
            rna_mask[i,:L_stack[i]] = out_dict["rna_mask"][None,]

            R_polymer_type[i,:L_stack[i]] = out_dict["R_polymer_type"][None,]

            interface_mask[i,:L_stack[i]] = out_dict["interface_mask"][None,]
            base_pair_mask[i,:L_stack[i]] = out_dict["base_pair_mask"][None,]
            base_pair_index[i,:L_stack[i]] = out_dict["base_pair_index"][None,]
            canonical_base_pair_mask[i,:L_stack[i]] = out_dict["canonical_base_pair_mask"][None,]
            canonical_base_pair_index[i,:L_stack[i]] = out_dict["canonical_base_pair_index"][None,]

            aligned_ppm[i,:L_stack[i]] = out_dict["aligned_ppm"][None,]
            ppm_mask[i,:L_stack[i]] = out_dict["ppm_mask"][None,]

            structure_paths.append(out_dict["structure_path"])
            assembly_ids.append(out_dict["assembly_id"])

        out_dict = {}

        out_dict["X"] = X.to(device) #[B, L, num_prot_atoms, 3]
        out_dict["X_m"] = X_m.to(device) #[B, L, num_prot_atoms]
        out_dict["mask"] = mask.to(device) #[B, L, num_prot_atoms]

        out_dict["S"] = S.long().to(device) #[B, L]
        out_dict["R_idx"] = R_idx.to(device) #[B, L]
        out_dict["chain_labels"] = chain_labels.to(device) #[B, L]

        out_dict["protein_mask"] = protein_mask.to(device)
        out_dict["dna_mask"] = dna_mask.to(device)
        out_dict["rna_mask"] = rna_mask.to(device)

        out_dict["R_polymer_type"] = R_polymer_type.to(device)

        out_dict["interface_mask"] = interface_mask.to(device)
        out_dict["base_pair_mask"] = base_pair_mask.to(device)
        out_dict["base_pair_index"] = base_pair_index.to(device)
        out_dict["canonical_base_pair_mask"] = canonical_base_pair_mask.to(device)
        out_dict["canonical_base_pair_index"] = canonical_base_pair_index.to(device)

        out_dict["aligned_ppm"] = aligned_ppm.to(device)
        out_dict["ppm_mask"] = ppm_mask.to(device)

        out_dict["structure_path"] = structure_paths
        out_dict["assembly_id"] = assembly_ids

        return out_dict

    else:
        return "pass"

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false

def loss_smoothed(S, 
                  log_probs, 
                  mask,
                  polymer_masks,
                  polymer_restype_masks,
                  polymer_restype_nums, 
                  weight=0.1,  
                  tokens=2000.0, 
                  num_letters=33,
                  ppm_mask=None,
                  aligned_ppm=None,):
    """ Negative log probabilities """
    protein_mask = polymer_masks["protein"]
    dna_mask = polymer_masks["dna"]
    rna_mask = polymer_masks["rna"]

    protein_restype_mask = polymer_restype_masks["protein"]
    dna_restype_mask = polymer_restype_masks["dna"]
    rna_restype_mask = polymer_restype_masks["rna"]
    all_polymer_restype_mask = protein_restype_mask + dna_restype_mask + rna_restype_mask

    S_onehot = torch.nn.functional.one_hot(S, num_letters).to(torch.float64)

    S_onehot[ppm_mask.bool()] = aligned_ppm[ppm_mask.bool()]

    label_smoothing_eps = protein_mask[:,:,None] * protein_restype_mask[None,None,:] * (weight / polymer_restype_nums["protein"]) + \
                          dna_mask[:,:,None] * dna_restype_mask[None,None,:] * (weight / polymer_restype_nums["dna"]) + \
                          rna_mask[:,:,None] * rna_restype_mask[None,None,:] * (weight / polymer_restype_nums["rna"])

    # Label smoothing
    S_onehot[:,:,all_polymer_restype_mask.bool()] *= (1 - weight)
    S_onehot += label_smoothing_eps

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / tokens #fixed 
    return loss, loss_av

def compute_canonical_base_pair_accuracy(log_probs, 
                                         canonical_base_pair_mask, 
                                         canonical_base_pair_index, 
                                         pdb_dataset):
    S_pred = torch.argmax(log_probs, -1) #[B,L]

    canonical_base_pair_pred = torch.gather(S_pred, 1, canonical_base_pair_index) #[B,L]

    canonical_base_pair_accuracy = torch.zeros_like(S_pred, dtype = torch.bool) #[B,L]
    for (res, canonical_base_pair_res) in pdb_dataset.na_canonical_base_pair_ints:
        canonical_base_pair_accuracy = \
            torch.logical_or(canonical_base_pair_accuracy, 
                             torch.logical_and(S_pred == res, canonical_base_pair_pred == canonical_base_pair_res))
    
    canonical_base_pair_accuracy = canonical_base_pair_accuracy.long()
    canonical_base_pair_accuracy = canonical_base_pair_accuracy * canonical_base_pair_mask

    return canonical_base_pair_accuracy

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

class DecLayerJ(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayerJ, self).__init__()
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
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,-1,h_E.size(-2),-1) #the only difference
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

class ProteinFeatures(nn.Module):
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
                 include_pred_na_N=1,
                 device=None):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()

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

class ProteinMPNN(nn.Module):
    def __init__(self, 
                 node_features=128, 
                 edge_features=128,
                 hidden_dim=128, 
                 num_encoder_layers=3, 
                 num_decoder_layers=3,
                 atom_dict=None,
                 restype_to_int=None,
                 polytype_to_int=None,
                 vocab=33,
                 num_letters=33, 
                 k_neighbors=32, 
                 protein_augment_eps=0.1, 
                 dna_augment_eps=0.1,
                 rna_augment_eps=0.1,
                 dropout=0.1, 
                 decode_protein_first=0, 
                 na_ref_atom="C1'",
                 include_pred_na_N=1,
                 device=None):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.decode_protein_first = decode_protein_first

        if restype_to_int is None:
            raise Exception("restype_to_int dictionary is necessary!")

        self.mask_token = restype_to_int["MAS"]

        self.features = ProteinFeatures(node_features, 
                                        edge_features,
                                        top_k=k_neighbors,
                                        atom_dict=atom_dict,
                                        polytype_to_int=polytype_to_int,
                                        protein_augment_eps=protein_augment_eps, 
                                        dna_augment_eps=dna_augment_eps, 
                                        rna_augment_eps=rna_augment_eps,
                                        na_ref_atom=na_ref_atom,
                                        include_pred_na_N=include_pred_na_N,
                                        device=device)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

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

    def forward(self, feature_dict):
        X = feature_dict["X"] 
        S = feature_dict["S"]
        mask = feature_dict["mask"]
        protein_mask = feature_dict["protein_mask"]
       
        device=X.device
        # Prepare node and edge embeddings
        V, E, E_idx = self.features(feature_dict)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            if self.training:
                h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend, use_reentrant = False)
            else:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)


        chain_M = mask #update chain_M to include missing regions

        if self.decode_protein_first:
            chain_M = chain_M.masked_fill(protein_mask.to(torch.bool), 0.0)

        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            if self.training:
                h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask, use_reentrant = False)
            else:
                h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        return log_probs, probs

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )
