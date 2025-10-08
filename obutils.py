from openbabel import openbabel
import numpy as np
import networkx as nx
import torch
import itertools
from typing import Dict
import random
import geometry

openbabel.obErrorLog.SetOutputLevel(0)


# ============================================================
# TODO: account for quasi-symmetric groups, e.g. C(=O)O, PO4, etc
def FindAutomorphisms(obmol : openbabel.OBMol,
                      heavy : bool = True,
                      maxmem : int = 2**20) -> torch.tensor:
    '''find automorphisms of a molecule
    
    Args:
    	obmol : the molecule for which to find the automorphisms
    	heavy : whether to use heavy atoms only
                (by default hydrogens are ignored)
        maxmem : max memory in bytes to be used by openbabel
                (default is ~1GB)
        
    Returns:
        [N,L] tensor storing automorphisms (!!!sorted!!!)
            N - number of automorphisms found
            L - number of atoms in the molecule
    '''

    L = obmol.NumAtoms()
    automorphs = openbabel.vvpairUIntUInt()
    mask = openbabel.OBBitVec(L)
    if heavy==False:
        mask.Negate()
    else:
        # mask out hydrogens if heavy==True
        for a in openbabel.OBMolAtomIter(obmol):
            if a.GetAtomicNum()!=1:
                mask.SetBitOn(a.GetIdx())
    try:
        openbabel.FindAutomorphisms(obmol,automorphs,mask,maxmem)
        permuts = torch.tensor(automorphs)
        permuts = torch.tensor([p[p[:,0].sort()[1],1].tolist() for p in permuts])

        # map permuts back to the all-atom molecule
        mask = torch.tensor([mask.BitIsSet(i+1) for i in range(L)])
        out = torch.arange(L)[None].repeat(permuts.shape[0],1)
        out[:,mask] = permuts
    
    except:
        # if the above failed, return the identity mapping
        return torch.arange(L)[None]
    
    return out


# ============================================================
def GetEquivalentHydrogens(obmol : openbabel.OBMol) -> torch.tensor:
    '''find all pairs of equivalent hydrogens
    (the ones attached to the same heavy atom)
    
    Args:
    	obmol : input moleclule
    
    Returns:
        [N,2] - pairs of equivalent hydrogens
    '''

    # bonds involving hydrogen atoms
    hbonds = [(a.GetIndex(),b.GetIndex()) 
              for a in openbabel.OBMolAtomIter(obmol) 
              for b in openbabel.OBAtomAtomIter(a) if b.GetAtomicNum()==1]

    # find groups of hydrogens attached to the same heavy atom
    groups = [[vi[1] for vi in v] for k,v in itertools.groupby(hbonds,key=lambda x : x[0])]

    # pairs of equivalent hydrogens
    #pairs = [pair for g in groups for pair in itertools.combinations(g,r=2)]
    pairs = [pair for g in groups for pair in itertools.product(g,repeat=2)]
    pairs = torch.tensor(pairs)

    return pairs


# ============================================================
def GetChirals(obmol : openbabel.OBMol,
               heavy : bool = True) -> torch.tensor:
    '''get all quadruples of atoms forming chiral centers

    Args:
        obmol : input molecule

    Returns:
        [Nx4] int tensor storing groups of atoms forming chiral centers
        the 4 columns are:
            0 - central atom
            1,2,3 - atoms connected to atom-0
        atom indices are ordered in such a way that the direction
        2->3 is clock-wise if seen from 1->0
        (equivalently, triple product of the three vectors
        [(o-i),(o-j),(o-k)] is positive)
    '''
    
    stereo = openbabel.OBStereoFacade(obmol)
    if stereo.NumTetrahedralStereo()<1:
        return torch.tensor([])
    
    chirals = []
    for i in range(obmol.NumAtoms()):
        if not stereo.HasTetrahedralStereo(i):
            continue
        si = stereo.GetTetrahedralStereo(i)
        config = si.GetConfig()

        o = config.center
        c = config.from_or_towards
        i,j,k = list(config.refs)

        chirals.extend([(o,c,i,j),
                        (o,c,j,k),
                        (o,c,k,i),
                        (o,k,j,i)])

    chirals = torch.tensor(chirals)
    chirals = chirals[(chirals<obmol.NumAtoms()).all(dim=-1)]
    
    # filter out hydrogens
    if heavy==True and chirals.shape[0]>0:
        hflag=torch.tensor([a.GetAtomicNum()==1 for a in openbabel.OBMolAtomIter(obmol)])
        chirals = chirals[~hflag[chirals].any(-1)]

    return chirals


# ============================================================
def GetPlanars(obmol : openbabel.OBMol,
               heavy : bool = True) -> torch.tensor:
    ''' '''

    # collect all sp2-hybridized atoms along with their neighbors
    sp2 = [(a.GetIndex(),*[b.GetIndex() for b in openbabel.OBAtomAtomIter(a)]) 
           for a in openbabel.OBMolAtomIter(obmol) if a.GetHyb()==2]
    
    # select centers with 3 neighbors only
    planars = torch.tensor([p for p in sp2 if len(p)==4])
    
    # filter out hydrogens
    if heavy==True and planars.shape[0]>0:
        hflag=torch.tensor([a.GetAtomicNum()==1 for a in openbabel.OBMolAtomIter(obmol)])
        planars = planars[~hflag[planars].any(-1)]

    return planars


# ============================================================
def GetTopology(obmol : openbabel.OBMol) -> Dict[str,torch.tensor]:
    ''' '''
    
    bonds = [(b.GetBeginAtom().GetIndex(),b.GetEndAtom().GetIndex()) 
             for b in openbabel.OBMolBondIter(obmol)]
    bondlen = [b.GetEquibLength() for b in openbabel.OBMolBondIter(obmol)]
    angles = [ang for ang in openbabel.OBMolAngleIter(obmol)]
    dihedrals = [dih for dih in openbabel.OBMolTorsionIter(obmol)]
    
    return{'bonds' : torch.tensor(bonds),
           'bondlen' : torch.tensor(bondlen),
           'angles' : torch.tensor(angles),
           'dihedrals' : torch.tensor(dihedrals), 
           'planars' : GetPlanars(obmol),
           'chirals' : GetChirals(obmol)}


# ============================================================
def ReduceHydrogens(obmol : openbabel.OBMol) -> torch.tensor:
    '''find mapping between full and reduced representations'''
    
    ijk = []
    heavy = [a for a in openbabel.OBMolAtomIter(obmol) if a.GetAtomicNum()!=1]
    for i,a in enumerate(heavy):
        ijk.append((i,0,a.GetIndex()))
        hydr = [h for h in openbabel.OBAtomAtomIter(a) if h.GetAtomicNum()==1]
        random.shuffle(hydr)
        for j,h in enumerate(hydr):
            ijk.append((i,j+1,h.GetIndex()))
    ijk.sort(key=lambda x : x[2])
    
    return torch.tensor(ijk)


# ============================================================
def GetFeatures1D(obmol : openbabel.OBMol) -> torch.tensor:
    '''get 1D features'''

    atoms = [obmol.GetAtom(i+1) for i in range(obmol.NumAtoms())]
    
    f1d = torch.tensor([[a.GetAtomicNum(),
                         a.GetFormalCharge(),
                         a.ExplicitHydrogenCount(),
                         a.GetHyb()] for a in atoms])
        
    return f1d


# ============================================================
def GetFeatures2D(obmol : openbabel.OBMol,
                  maxpath : int=8) -> torch.tensor:
    '''get 2D features'''

    # connectivity graph
    bonds = list(openbabel.OBMolBondIter(obmol))
    N = obmol.NumAtoms()
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from([(b.GetBeginAtom().GetIndex(),b.GetEndAtom().GetIndex()) for b in bonds])
        
    # 2d features:
    #   (1) is aromatic
    #   (2) is in ring
    #   (3) bond order
    #   (4) bond separation
    f2d = torch.zeros((N,N,4), dtype=torch.long)
    if len(bonds)<1:
        return f2d

    # aromatic, in ring, order
    i,j = np.array(G.edges).T
    f2d[i,j,:3] = torch.tensor([(b.IsAromatic(),b.IsInRing(),b.GetBondOrder()) for b in bonds])
    f2d[j,i,:3] = f2d[i,j,:3]

    # bond separation
    paths = dict(nx.all_pairs_shortest_path_length(G,cutoff=maxpath))
    paths = [(i,j,vij) for i,vi in paths.items() for j,vij in vi.items()]
    i,j,v = torch.tensor(paths).T
    f2d[i,j,3] = v

    return f2d


# ============================================================
class OBMolFeaturizer:
    #'''
    def __init__(self,
                 maxpath : int = 8,
                 maxcharge : int = 6,
                 maxhyb : int = 24,
                 maxhydr : int = 12):
        
        #self.with_h = explicit_hydrogens
        self.maxpath = maxpath
        self.maxcharge = maxcharge
        self.maxhyb = maxhyb
        self.maxhydr = maxhydr
        
        self.dims1d = (118,maxcharge*2,maxhydr,maxhyb+1)
        self.dims2d = (2,2,4,maxpath+1)
        
        # parse elements' electronic structure table
        spdf = [
            ('1s',2), ('2s',2), ('2p',6), ('3s',2), ('3p',6),
            ('4s',2), ('3d',10), ('4p',6), ('5s',2), ('4d',10), 
            ('5p',6), ('6s',2), ('4f',14), ('5d',10), ('6p',6),
            ('7s',2), ('5f',14), ('6d',10), ('7p',6)
        ]
        self.econf = {}
        # TODO: make path relative
        for l in open('/home/aivan/git/chemnet/arch.22-10-28/data/elements.txt','r').readlines():
            num,element,shell_str = l.strip().split('\t')
            shell = {s[:2]:int(s[2:]) for s in shell_str.split()}
            shell = [[1]*shell[k]+[0]*(v-shell[k]) if k in shell.keys() else [0]*v for k,v in spdf]
            shell = torch.tensor([si for s in shell for si in s]).float()
            self.econf[int(num)] = shell


    def GetFeatures1D(self, obmol : openbabel.OBMol) -> torch.tensor:
        '''get 1D features'''
        
        atoms = list(openbabel.OBMolAtomIter(obmol))
        idx = torch.tensor([a.GetIndex() for a in atoms])

        '''
        # element embedding
        element = torch.stack([self.econf[a.GetAtomicNum()] for a in atoms], dim=0)
        
        # charge embedding
        qmax = self.maxcharge
        EmbedCharge = lambda q : [1]*abs(q)+[0]*(2*qmax-abs(q)) if q<0 else [0]*qmax+[1]*q+[0]*(qmax-q)
        charge = torch.tensor([EmbedCharge(a.GetFormalCharge()) for a in atoms])

        # number of hydrogens
        hmax = self.maxhydr
        EmbedHydr = lambda h : [1]*h+[0]*(hmax-h) if h<hmax else [1]*hmax
        hydr = torch.tensor([EmbedHydr(a.ExplicitHydrogenCount()) for a in atoms])

        # hybridization
        hmax = self.maxhyb
        EmbedHyb = lambda h : torch.eye(hmax+1)[h] if h<hmax else torch.zeros((hmax+1))
        hyb = torch.stack([EmbedHyb(a.GetHyb()) for a in atoms], dim=0)
        
        # stack all features together
        f1d = torch.cat([element,charge,hydr,hyb], dim=-1)
        '''
        f1d = torch.tensor([[a.GetAtomicNum(),
                             a.GetFormalCharge(),
                             a.ExplicitHydrogenCount(),
                             a.GetHyb()] for a in atoms])
        
        # make sure features' order is consistent with atom indices
        f1d[idx] = f1d.clone()
        
        return f1d
    
        
    def GetFeatures2D(self, obmol : openbabel.OBMol) -> torch.tensor:
        '''get 2D features'''

        # connectivity graph
        bonds = list(openbabel.OBMolBondIter(obmol))
        N = obmol.NumAtoms()
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from([(b.GetBeginAtom().GetIndex(),b.GetEndAtom().GetIndex()) for b in bonds])
        
        # 2d features:
        #   (1) is aromatic
        #   (2) is in ring
        #   (3) bond order
        #   (4) bond separation
        if len(bonds)<1:
            return torch.zeros((N,N,self.NumFeatures2D()), dtype=float)

        f2d = torch.zeros((4,N,N), dtype=torch.long)

        # aromatic, in ring, order
        i,j = np.array(G.edges).T
        f2d[:3,i,j] = torch.tensor([(b.IsAromatic(),b.IsInRing(),b.GetBondOrder()) for b in bonds]).T
        f2d[:3,j,i] = f2d[:3,i,j]

        # bond separation
        paths = dict(nx.all_pairs_shortest_path_length(G,cutoff=self.maxpath))
        paths = [(i,j,vij) for i,vi in paths.items() for j,vij in vi.items()]
        i,j,v = torch.tensor(paths).T
        f2d[3,i,j] = v

        # apply one-hot encoding
        f2d = torch.cat([torch.nn.functional.one_hot(f,d)
                         for f,d in zip(f2d,self.dims2d)], dim=-1).float()
        
        return f2d

    
    def NumFeatures1D(self) -> int:
        '''get the number of 1D features'''
        return sum(self.dims1d)
        

    def NumFeatures2D(self) -> int:
        '''get the number of 2D features'''
        return sum(self.dims2d)


    def ReduceHydrogens(self, 
                        obmol : openbabel.OBMol,
                        f1d : torch.tensor,
                        f2d : torch.tensor) -> Dict[str,torch.tensor]:
        '''move hydrogen atoms to the adjacent heavy atom
        
        Args:
            obmol : input molecule with hydrogens
            f1d : [L,NumFeatures1D] - 1D features
            f2d : [L,L,NumFeatures2D] - 2D features

        Returns:
            {
                'xyz' : xyz, [Lheavy,Nhydr+1,3] - heavy atoms coordinates followed by the coordinates of the adjacent hydrogens
                'f1d' : f1d, [Lheavy,NumFeatures1D] - 1D features with hydrogens omitted
                'f2d' : f2d, [Lheavy,Lheavy,NumFeatures2D] - 2D features with hydrogens omitted
                'ijk' : ijk, [L,3] - mapping between reduced and full representations
                'observed' : observed, [L] - mask to indicate physically observed atoms
                'heavy' : heavy, [L] - heavy atoms mask
            }
        
        '''

        heavy = [a for a in openbabel.OBMolAtomIter(obmol) if a.GetAtomicNum()!=1]
        
        ijk = []
        xyz = torch.full((len(heavy),self.maxhydr+1,3), np.nan)
        observed = torch.zeros((obmol.NumAtoms())).bool()
        for i,a in enumerate(heavy):
            xyz[i,0] = torch.tensor([a.x(),a.y(),a.z()])
            observed[a.GetIndex()] = True
            ijk.append((i,0,a.GetIndex()))
            
            hydr = [h for h in openbabel.OBAtomAtomIter(a) if h.GetAtomicNum()==1]
            random.shuffle(hydr)
            for j,h in enumerate(hydr[:self.maxhydr]):
                xyz[i,j+1] = torch.tensor([h.x(),h.y(),h.z()])
                observed[h.GetIndex()] = True
                ijk.append((i,j+1,h.GetIndex()))
        
        heavy_mask = torch.tensor([a.GetAtomicNum()!=1 for a in openbabel.OBMolAtomIter(obmol)])

        return {
            'xyz' : xyz,
            'f1d' : f1d[heavy_mask],
            'f2d' : f2d[heavy_mask][:,heavy_mask],
            'ijk' : torch.tensor(ijk),
            'observed' : observed,
            'heavy' : heavy_mask
        }
