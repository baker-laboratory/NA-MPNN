import sys
import json
import gzip
import re
import copy
import collections
import random
from openbabel import openbabel
import itertools
from typing import Dict,List
import numpy as np
import networkx as nx
import torch
import pdbx
from pdbx.reader.PdbxReader import PdbxReader
from pdbx.reader.PdbxContainers import DataCategory

#sys.path.insert(0, "/home/justas/projects/lab_github/chemnet/arch.22-10-28")
import obutils
import geometry

# TODO:
#   1) overlapping ligands/residues


# ============================================================
Atom = collections.namedtuple('Atom', [
    'name',
    'xyz', # Cartesian coordinates of the atom
    'occ', # occupancy
    'bfac', # B-factor
    'leaving', # boolean flag to indicate whether the atom leaves the molecule upon bond formation
    'leaving_group', # list of atoms which leave the molecule if a bond with this atom is formed
    'parent', # neighboring heavy atom this atom is bonded to
    'element', # atomic number (1..118)
    'metal', # is this atom a metal? (bool)
    'charge', # atomic charge (int)
    'hyb', # hybridization state (int)
    'nhyd', # number of hydrogens
    'hvydeg', # heavy atom degree
    'align', # atom name alignment offset in PDB atom field
    'hetero'
])

Bond = collections.namedtuple('Bond', [
    'a','b', # names of atoms forming the bond (str)
    'aromatic', # is the bond aromatic? (bool)
    'in_ring', # is the bond in a ring? (bool)
    'order', # bond order (int)
    'intra', # is the bond intra-residue? (bool)
    'length' # reference bond length from openbabel (float)
])

Residue = collections.namedtuple('Residue', [
    'name',
    'atoms',
    'bonds',
    'automorphisms',
    'chirals',
    'planars',
    'alternatives'
])

Chain = collections.namedtuple('Chain', [
    'id',
    'type',
    'sequence',
    'atoms',
    'bonds',
    'chirals',
    'planars',
    'automorphisms'
])


# ============================================================
def ParsePDBLigand(cifname : str) -> Dict:
    '''Parse a single molecule from the PDB-Ligands set
    '''

    data = []
    with open(cifname,'r') as cif:
        reader = PdbxReader(cif)
        reader.read(data)
    data = data[0]
    chem_comp_atom = data.getObj('chem_comp_atom')
    rows = chem_comp_atom.getRowList()

    # parse atom names
    idx = chem_comp_atom.getIndex('atom_id')
    atom_id = np.array([r[idx] for r in rows])
    
    # parse element symbols
    idx = chem_comp_atom.getIndex('type_symbol')
    symbol = np.array([r[idx] for r in rows])

    # parse leaving flags
    idx = chem_comp_atom.getIndex('pdbx_leaving_atom_flag')
    leaving = [r[idx] for r in rows]
    leaving = np.array([True if flag=='Y' else False for flag in leaving], dtype=bool)

    # atom name alignment offset in PDB atom field
    idx = chem_comp_atom.getIndex('pdbx_align')
    pdbx_align = np.array([int(r[idx]) for r in rows])
    
    # parse xyz coordinates
    i = chem_comp_atom.getIndex('model_Cartn_x')
    j = chem_comp_atom.getIndex('model_Cartn_y')
    k = chem_comp_atom.getIndex('model_Cartn_z')
    xyz = [(r[i],r[j],r[k]) for r in rows]
    xyz = np.array([[float(c) if c!='?' else np.nan for c in p] for p in xyz])

    out = {'atom_id' : atom_id,
           'leaving' : leaving,
           'symbol' : symbol,
           'pdbx_align' : pdbx_align,
           'xyz' : xyz}

    return out


# ============================================================
class CIFParser:
    
    def __init__(self, skip_res : List[str] = None, randomize_nmr_model = 0):
        
        self.randomize_nmr_model = randomize_nmr_model

        # parse pre-compiled library of all residues observed in the PDB
        with gzip.open('/home/akubaney/projects/na_mpnn/data/datasets/rcsb_cif/ligands.json.gz','rt') as file:
            self.mols = json.load(file)
            
            # residues to be ignored during parsing are
            # deleted form the library
            if skip_res is not None:
                for res in skip_res:
                    if res in self.mols.keys():
                        del self.mols[res]
            
        with open('/home/akubaney/projects/na_mpnn/data/datasets/rcsb_cif/elements.txt','r') as f:
            self.i2a = [l.strip().split()[:2] for l in f.readlines()]
            self.i2a = {int(i):a for i,a in self.i2a}
    

    def getRes(self,resname : str) -> Residue:
        '''get a residue from the library; residues are loaded dynamically'''
        
        res = self.mols.get(resname)
        if res is None:
            return res
        
        if 'res' not in res.keys():
            res['res'] = self.parseLigand(sdfstring=res['sdf'],
                                          atom_id=res['atom_id'],
                                          leaving=res['leaving'],
                                          pdbx_align=res['pdbx_align'])
        return res
        
        
    @staticmethod
    def getLeavingAtoms(a,leaving,s):
        for b in openbabel.OBAtomAtomIter(a):
            if leaving[b.GetIndex()]==True:
                if b.GetIndex() not in s:
                    s.append(b.GetIndex())
                    CIFParser.getLeavingAtoms(b,leaving,s)


    @staticmethod
    def getLeavingAtoms2(aname, G):

        leaving_group = set()
    
        if G.nodes[aname]['leaving']==True:
            return []

        for m in G.neighbors(aname):
            if G.nodes[m]['leaving']==False:
                continue
            leaving_group.update({m})
            H = G.subgraph(set(G.nodes)-{m})
            ccs = list(nx.connected_components(H))
            if len(ccs)>1:
                for cc in ccs:
                    if aname not in cc:
                        leaving_group.update(cc)

        return list(leaving_group)


    @staticmethod
    def parseLigand(sdfstring : str,
                    atom_id : List[str],
                    leaving : List[bool],
                    pdbx_align : List[int],) -> Residue:

        # create molecule from the sdf string
        obmol = openbabel.OBMol()
        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat("sdf")
        obConversion.ReadString(obmol,sdfstring)
        #obmol.DeleteHydrogens()
        
        # correct for pH to get some charged groups
        obmol_ph = openbabel.OBMol(obmol)
        obmol_ph.CorrectForPH()
        obmol_ph.DeleteHydrogens()
        ha_iter = openbabel.OBMolAtomIter(obmol_ph)                
        
        # get atoms and their features
        atoms = {}
        for aname,aleaving,align,a in zip(atom_id,leaving,pdbx_align,openbabel.OBMolAtomIter(obmol)):

            '''
            leaving_group = []
            if aleaving==False:
                CIFParser.getLeavingAtoms(a,leaving,leaving_group)
                leaving_group = [atom_id[i] for i in leaving_group]
            '''
            
            # parent heavy atoms
            parent = None
            for b in openbabel.OBAtomAtomIter(a):
                if b.GetAtomicNum()>1:
                    parent = atom_id[b.GetIndex()]
            
            charge = a.GetFormalCharge()
            nhyd = a.ExplicitHydrogenCount()
            if a.GetAtomicNum()>1:
                ha = next(ha_iter)
                charge = ha.GetFormalCharge()
                nhyd = ha.GetTotalDegree()-ha.GetHvyDegree()
            
            atoms[aname] = Atom(name=aname,
                                xyz=[0.0,0.0,0.0],
                                occ=0.0,
                                bfac=0.0,
                                leaving=aleaving,
                                #leaving_group=leaving_group,
                                leaving_group=[],
                                parent=parent,
                                element=a.GetAtomicNum(),
                                metal=a.IsMetal(),
                                charge=charge,
                                hyb=a.GetHyb(),
                                nhyd=nhyd,
                                align=align,
                                hvydeg=a.GetHvyDegree(),
                                hetero=False)
        
        # get bonds and their features
        bonds = []
        for b in openbabel.OBMolBondIter(obmol):
            bonds.append(Bond(a=atom_id[b.GetBeginAtom().GetIndex()],
                              b=atom_id[b.GetEndAtom().GetIndex()],
                              aromatic=b.IsAromatic(),
                              in_ring=b.IsInRing(),
                              order=b.GetBondOrder(),
                              intra=True,
                              length=b.GetLength()))

        # get automorphisms only retaining atoms with alternative mappings
        automorphisms = obutils.FindAutomorphisms(obmol, heavy=True)
        mask = (automorphisms[:1]==automorphisms).all(dim=0)
        automorphisms = automorphisms[:,~mask]
        
        # skip automorphisms which include leaving atoms
        if automorphisms.shape[0]>1:
            mask = torch.tensor(leaving)[automorphisms].any(dim=1)
            automorphisms = automorphisms[~mask]
        
        # get chirals and planars
        chirals = obutils.GetChirals(obmol, heavy=True)
        planars = obutils.GetPlanars(obmol, heavy=True)

        # add leaving groups to atoms
        G = nx.Graph()
        G.add_nodes_from([(a.name,{'leaving':a.leaving}) for a in atoms.values()])
        G.add_edges_from([(bond.a,bond.b) for bond in bonds])
        for k,v in atoms.items():
            atoms[k] = v._replace(leaving_group=CIFParser.getLeavingAtoms2(k,G))
        
        # put everything into a residue
        anames = np.array(atom_id)
        R = Residue(name=obmol.GetTitle(),
                    atoms=atoms,
                    bonds=bonds,
                    automorphisms=anames[automorphisms].tolist(),
                    chirals=anames[chirals].tolist(),
                    planars=anames[planars].tolist(),
                    alternatives=set())

        return R

    
    @staticmethod
    def parseOperationExpression(expression : str) -> List:
        '''a function to parse _pdbx_struct_assembly_gen.oper_expression 
        into individual operations'''

        expression = expression.strip('() ')
        operations = []
        for e in expression.split(','):
            e = e.strip()
            pos = e.find('-')
            if pos>0:
                start = int(e[0:pos])
                stop = int(e[pos+1:])
                operations.extend([str(i) for i in range(start,stop+1)])
            else:
                operations.append(e)

        return operations


    @staticmethod
    def parseAssemblies(data : pdbx.reader.PdbxContainers.DataContainer) -> Dict:
        '''parse biological assembly data'''
        
        assembly_data = data.getObj("pdbx_struct_assembly")
        assembly_gen = data.getObj("pdbx_struct_assembly_gen")
        oper_list = data.getObj("pdbx_struct_oper_list")

        if (assembly_data is None) or (assembly_gen is None) or (oper_list is None):
            return {}

        # save all basic transformations in a dictionary
        opers = {}
        for k in range(oper_list.getRowCount()):
            key = oper_list.getValue("id", k)
            val = np.eye(4)
            for i in range(3):
                val[i,3] = float(oper_list.getValue("vector[%d]"%(i+1), k))
                for j in range(3):
                    val[i,j] = float(oper_list.getValue("matrix[%d][%d]"%(i+1,j+1), k))
            opers.update({key:val})

        chains,ids = [],[]
        xforms = []
        #details,method = [],[]

        for index in range(assembly_gen.getRowCount()):

            # Retrieve the assembly_id attribute value for this assembly
            assemblyId = assembly_gen.getValue("assembly_id", index)
            ids.append(assemblyId)

            # Retrieve the operation expression for this assembly from the oper_expression attribute	
            oper_expression = assembly_gen.getValue("oper_expression", index)

            oper_list = [CIFParser.parseOperationExpression(expression) 
                         for expression in re.split('\(|\)', oper_expression) if expression]

            # chain IDs which the transform should be applied to
            chains.append(assembly_gen.getValue("asym_id_list", index).split(','))

            #index_asmb = min(index,assembly_data.getRowCount()-1)
            #details.append(assembly_data.getValue("details", index_asmb))
            #method.append(assembly_data.getValue("method_details", index_asmb))

            if len(oper_list)==1:
                xforms.append(np.stack([opers[o] for o in oper_list[0]]))
            elif len(oper_list)==2:
                xforms.append(np.stack([opers[o1]@opers[o2] 
                                        for o1 in oper_list[0] 
                                        for o2 in oper_list[1]]))
            else:
                print('Error in processing assembly')           
                return xforms

        # return xforms as a dict {asmb_id:[(chain_id,xform[4,4])]}
        #out = dict.fromkeys(set(ids),[])
        out = {i:[] for i in set(ids)}
        for key,c,x in zip(ids,chains,xforms):
            out[key].extend(itertools.product(c,x))
            
        return out


    def parse(self, filename : str) -> Dict:
        
        ########################################################
        # 0. read a .cif file
        ########################################################
        data = []
        if filename.endswith('.gz'):
            with gzip.open(filename,'rt') as cif:
                reader = PdbxReader(cif)
                reader.read(data)
        else:
            with open(filename,'r') as cif:
                reader = PdbxReader(cif)
                reader.read(data)
        data = data[0]


        ########################################################
        # 1. parse mappings of modified residues to their 
        #    standard counterparts
        ########################################################
        pdbx_struct_mod_residue = data.getObj('pdbx_struct_mod_residue')
        if pdbx_struct_mod_residue is None:
            modres = {}
        else:
            modres = {(r[pdbx_struct_mod_residue.getIndex('label_comp_id')],
                       r[pdbx_struct_mod_residue.getIndex('parent_comp_id')])
                      for r in pdbx_struct_mod_residue.getRowList()}
            modres = {k:v for k,v in modres if k!=v}


        ########################################################
        # 2. parse polymeric chains
        ########################################################
        pdbx_poly_seq_scheme = data.getObj('pdbx_poly_seq_scheme')
        chains = {}
        if pdbx_poly_seq_scheme is not None:
            
            # establish mapping asym_id <--> (entity_id,pdb_strand_id)
            chains = {
                r[pdbx_poly_seq_scheme.getIndex('asym_id')]: {
                    'entity_id' : r[pdbx_poly_seq_scheme.getIndex('entity_id')],
                    'pdb_strand_id' : r[pdbx_poly_seq_scheme.getIndex('pdb_strand_id')]}
                for r in pdbx_poly_seq_scheme.getRowList() }
            
            # parse canonical 1-letter sequences
            entity_poly = data.getObj('entity_poly')
            if entity_poly is not None:
                for r in entity_poly.getRowList():
                    entity_id = r[entity_poly.getIndex('entity_id')]
                    type_ = r[entity_poly.getIndex('type')]
                    seq = r[entity_poly.getIndex('pdbx_seq_one_letter_code_can')].replace('\n','')
                    for k,v in chains.items():
                        if v['entity_id']==entity_id:
                            v['type'] = r[entity_poly.getIndex('type')]
                            v['seq'] = seq

            # parse residues that are actually present in the polymer
            entity_poly_seq = data.getObj('entity_poly_seq')
            residues = [(r[entity_poly_seq.getIndex('entity_id')],
                         r[entity_poly_seq.getIndex('num')],
                         r[entity_poly_seq.getIndex('mon_id')],
                         r[entity_poly_seq.getIndex('hetero')] in {'y','yes'}) 
                        for r in entity_poly_seq.getRowList()]
            for entity_id,res in itertools.groupby(residues, key=lambda x : x[0]):
                res = [resi[1:] for resi in list(res)]
                
                # when there are alternative residues at the same position
                # pick the one which occurs first
                res = {k:Residue(*self.getRes(next(v)[1])['res'][:-1],alternatives=set([vi[1] for vi in v]))
                       for k,v in itertools.groupby(res, key=lambda x : x[0])}

                for k,v in chains.items():
                    if v['entity_id']==entity_id:
                        v['res'] = {k:copy.deepcopy(v) for k,v in res.items()}


        ########################################################
        # 3. parse non-polymeric molecules
        ########################################################
        
        # parse from HETATM
        atom_site = data.getObj('atom_site')
        nonpoly_res = [(r[atom_site.getIndex('label_asym_id')],
                        r[atom_site.getIndex('label_entity_id')],
                        r[atom_site.getIndex('auth_asym_id')],
                        r[atom_site.getIndex('auth_seq_id')], # !!! this is not necessarily an integer number, per mmcif specifications !!!
                        r[atom_site.getIndex('auth_comp_id')]) 
                       for r in atom_site.getRowList() 
                       if r[atom_site.getIndex('group_PDB')]=='HETATM' and r[atom_site.getIndex('label_asym_id')] not in chains.keys()]
        nonpoly_res = [r for r in nonpoly_res if r[0] not in chains.keys()]
        nonpoly_chains = {r[0]:{'entity_id':r[1], 'pdb_strand_id':r[2],'type':'nonpoly','res':{}} for r in nonpoly_res}
        for r in nonpoly_res:
            #res = self.mols.get(r[4])
            res = self.getRes(r[4])
            if res is not None:
                res = res['res']
            nonpoly_chains[r[0]]['res'][r[3]] = res
        for v in nonpoly_chains.values():
            v['res'] = {k2:copy.deepcopy(v2) for k2,v2 in v['res'].items()}
                
        chains.update(nonpoly_chains)


        ########################################################
        # 4. populate residues with coordinates
        ########################################################
        
        i = {k:atom_site.getIndex(val) for k,val in [('hetero', 'group_PDB'),
                                                     ('symbol', 'type_symbol'),
                                                     ('atm', 'label_atom_id'), # atom name
                                                     ('res', 'label_comp_id'), # residue name (3-letter)
                                                     ('chid', 'label_asym_id'), # chain ID
                                                     ('num', 'label_seq_id'), # sequence number
                                                     ('num_author', 'auth_seq_id'), # sequence number a
                                                     ('alt', 'label_alt_id'), # alternative location ID
                                                     ('x', 'Cartn_x'), # xyz coords
                                                     ('y', 'Cartn_y'),
                                                     ('z', 'Cartn_z'),
                                                     ('occ', 'occupancy'), # occupancy
                                                     ('bfac', 'B_iso_or_equiv'), # B-factors 
                                                     ('model', 'pdbx_PDB_model_num') # model number (for multi-model PDBs, e.g. NMR)
                                                    ]}
        
        row_list = atom_site.getRowList()
        last_model = row_list[-1][-1]
        if last_model.isnumeric():
            num_models = int(last_model)
            model_to_load = np.random.randint(num_models) + 1
        else:
            num_models = None
            model_to_load = None

        for r in atom_site.getRowList():

            hetero, symbol, atm, res, chid, num, num_author, alt, x, y, z, occ, bfac, model = \
                (t(r[i[k]]) for k,t in (('hetero',str), ('symbol',str), ('atm',str), ('res',str), ('chid',str), 
                                        ('num',str), ('num_author',str), ('alt',str),
                                        ('x',float), ('y',float), ('z',float), 
                                        ('occ',float), ('bfac',float), ('model',int)))
            
            if self.randomize_nmr_model and model_to_load is not None:
                if "NMR" in data.getObj('exptl').getValue('method',0).replace(' ','_') and model != model_to_load:
                    continue
            
            # we use author assigned residue numbers for non-polymeric chains
            if chains[chid]['type']=='nonpoly':
                num = num_author
            if num=='.': # !!! fixes 1ZY8 is which FAD ligand is assigned to a polypeptide chain O !!!
                continue
            residue = chains[chid]['res'][num]
            # skip if residue is not in the library
            if residue is not None and residue.name==res:
                # if any heavy atom in a residue cannot be matched
                # then mask the whole residue
                if atm not in residue.atoms.keys():
                    if symbol!='H' and symbol!='D':
                        #print('cannot map atom:', filename, res, atm, r)
                        chains[chid]['res'][num] = None
                    continue
                atom = residue.atoms[atm]
                if occ>atom.occ:
                    '''
                    atom = Atom(name=atom.name,
                                xyz=[x,y,z],
                                occ=occ,
                                leaving=atom.leaving,
                                leaving_group=atom.leaving_group,
                                parent=atom.parent,
                                element=atom.element,
                                charge=atom.charge,
                                hyb=atom.hyb,
                                hetero=(hetero=='HETATM'))
                    residue.atoms[atm] = atom
                    '''
                    residue.atoms[atm] = atom._replace(xyz=[x,y,z], 
                                                       occ=occ,
                                                       bfac=bfac,
                                                       hetero=(hetero=='HETATM'))
            

        ########################################################
        # 5. parse covalent connections
        ########################################################
        
        struct_conn = data.getObj('struct_conn')
        if struct_conn is not None:
            covale = [(r[struct_conn.getIndex('ptnr1_label_asym_id')],
                       r[struct_conn.getIndex('ptnr1_label_seq_id')],
                       r[struct_conn.getIndex('ptnr1_auth_seq_id')],
                       r[struct_conn.getIndex('ptnr1_label_comp_id')],
                       r[struct_conn.getIndex('ptnr1_label_atom_id')],
                       r[struct_conn.getIndex('ptnr2_label_asym_id')],
                       r[struct_conn.getIndex('ptnr2_label_seq_id')],
                       r[struct_conn.getIndex('ptnr2_auth_seq_id')],
                       r[struct_conn.getIndex('ptnr2_label_comp_id')],
                       r[struct_conn.getIndex('ptnr2_label_atom_id')])
                      for r in struct_conn.getRowList() if r[struct_conn.getIndex('conn_type_id')]=='covale']
            F = lambda x : x[2] if chains[x[0]]['type']=='nonpoly' else x[1]
            # here we skip intra-residue covalent bonds assuming that
            # they are properly handled by parsing from the residue library
            covale = [((c[0],F(c[:4]),c[3],c[4]),(c[5],F(c[5:]),c[8],c[9])) 
                      for c in covale if c[:4]!=c[5:8]]
                        
        else:
            covale = []


        ########################################################
        # 6. build connected chains
        ########################################################
        return_chains = {}
        for chain_id,chain in chains.items():
                        
            residues = list(chain['res'].items())
            atoms,bonds,skip_atoms = [],[],[]
            
            # (a) add inter-residue connections in polymers
            if 'polypept' in chain['type']:
                ab = ('C','N')
            elif 'polyribo' in chain['type'] or 'polydeoxyribo' in chain['type']:
                ab = ("O3'",'P')
            else:
                ab = ()

            if len(ab)>0:
                for ra,rb in zip(residues[:-1],residues[1:]):
                    # check for skipped residues (the ones failed in step 4)
                    if ra[1] is None or rb[1] is None:
                        continue
                    a = ra[1].atoms.get(ab[0])
                    b = rb[1].atoms.get(ab[1])
                    if a is not None and b is not None:
                        bonds.append(Bond(
                            a=(chain_id,ra[0],ra[1].name,a.name),
                            b=(chain_id,rb[0],rb[1].name,b.name),
                            aromatic=False,
                            in_ring=False,
                            order=1, # !!! we assume that all inter-residue bonds are single !!!
                            intra=False,
                            length=1.5 # !!! this is a rough guesstimate !!!
                        ))
                        skip_atoms.extend([(chain_id,ra[0],ra[1].name,ai) for ai in a.leaving_group])
                        skip_atoms.extend([(chain_id,rb[0],rb[1].name,bi) for bi in b.leaving_group])

            # (b) add connections parsed from mmcif's struct_conn record
            for ra,rb in covale:
                a = b = None
                if ra[0]==chain_id and chain['res'][ra[1]] is not None and chain['res'][ra[1]].name==ra[2]:
                    a = chain['res'][ra[1]].atoms[ra[3]]
                    #if a.charge!=0 and 'polypept' not in chain['type']:
                    #    print(filename, chain['type'], ra,rb, a)
                    skip_atoms.extend([(chain_id,*ra[1:3],ai) for ai in a.leaving_group])
                if rb[0]==chain_id and chain['res'][rb[1]] is not None and chain['res'][rb[1]].name==rb[2]:
                    b = chain['res'][rb[1]].atoms[rb[3]]
                    #if b.charge!=0  and 'polypept' not in chain['type']:
                    #    print(filename, chain['type'], ra,rb, b)
                    skip_atoms.extend([(chain_id,*rb[1:3],bi) for bi in b.leaving_group])
                if a is not None and b is not None:
                    bonds.append(Bond(
                        a=(chain_id,*ra[1:3],a.name),
                        b=(chain_id,*rb[1:3],b.name),
                        aromatic=False,
                        in_ring=False,
                        order=1, # !!! we assume that all inter-residue bonds are single !!!
                        intra=False,
                        length=1.5 # !!! this is a rough guesstimate !!!
                    ))
                    
            # (c) collect atoms
            skip_atoms = set(skip_atoms)
            atoms = {(chain_id,r[0],r[1].name,aname):a for r in residues if r[1] is not None
                     for aname,a in r[1].atoms.items()}
            atoms = {aname:a._replace(name=aname) for aname,a in atoms.items() if aname not in skip_atoms}
            
            # (d) collect intra-residue bonds
            '''
            bonds_intra = [Bond(a=(chain_id,r[0],r[1].name,bond.a),
                                b=(chain_id,r[0],r[1].name,bond.b),
                                aromatic=bond.aromatic,
                                in_ring=bond.in_ring,
                                order=bond.order)
                           for r in residues if r[1] is not None
                           for bond in r[1].bonds]
            '''
            bonds_intra = [bond._replace(a=(chain_id,r[0],r[1].name,bond.a),
                                         b=(chain_id,r[0],r[1].name,bond.b))
                           for r in residues if r[1] is not None
                           for bond in r[1].bonds]
            bonds_intra = [bond for bond in bonds_intra 
                           if bond.a not in skip_atoms and \
                           bond.b not in skip_atoms]

            bonds.extend(bonds_intra)
            
            # (e) double check whether bonded atoms actually exist:
            #     some could be part of the skip_atoms set and thus removed
            bonds = [bond for bond in bonds if bond.a in atoms.keys() and bond.b in atoms.keys()]
            bonds = list(set(bonds))
            
            # (f) fix charges and hydrogen counts for cases when
            #     charged atoms are connected by an inter-residue bond
            '''
            for bond in bonds:
                if bond.intra==False:
                    a = atoms[bond.a]
                    b = atoms[bond.b]

                    # nitrogen (involved in a peptide bond)
                    if a.element==7 and a.charge==1 and a.hyb==3 and a.nhyd==3:
                        atoms[bond.a] = a._replace(charge=0, hyb=2, nhyd=1)
                    if b.element==7 and b.charge==1 and b.hyb==3 and b.nhyd==3:
                        atoms[bond.b] = b._replace(charge=0, hyb=2, nhyd=1)
            
                    # oxygen (nucletides)
                    if a.element==8 and a.charge==-1 and a.hyb==3 and a.nhyd==0:
                        atoms[bond.a] = a._replace(charge=0)
                    if b.element==8 and b.charge==-1 and b.hyb==3 and b.nhyd==0:
                        atoms[bond.b] = b._replace(charge=0)
            '''
            
            # (g) relabel chirals, planars and automorphisms 
            #     to include residue indices and names
            chirals = [[(chain_id,r[0],r[1].name,c) for c in chiral] 
                       for r in residues if r[1] is not None for chiral in r[1].chirals]
            
            planars = [[(chain_id,r[0],r[1].name,c) for c in planar] 
                       for r in residues if r[1] is not None for planar in r[1].planars]
            
            #automorphisms = [[[(chain_id,r[0],r[1].name,a,len(r[1].automorphisms)) 
            automorphisms = [[[(chain_id,r[0],r[1].name,a) 
                               for a in auto] for auto in r[1].automorphisms] 
                             for r in residues if r[1] is not None and len(r[1].automorphisms)>1]

            chirals = [c for c in chirals if all([ci in atoms.keys() for ci in c])]
            planars = [c for c in planars if all([ci in atoms.keys() for ci in c])]

            if len(atoms)>0:
                return_chains[chain_id] = Chain(id=chain_id,
                                                type=chain['type'],
                                                sequence=chain.get('seq'),
                                                atoms=atoms,
                                                bonds=bonds,
                                                chirals=chirals,
                                                planars=planars,
                                                automorphisms=automorphisms)

                
        ########################################################
        # 6. parse assemblies
        ########################################################
        asmb = self.parseAssemblies(data)
        asmb = {k:[vi for vi in v if vi[0] in return_chains.keys()]
                for k,v in asmb.items()}

        
        # convert covalent links to Bonds
        covale = [Bond(a=c[0],
                       b=c[1],
                       aromatic=False,
                       in_ring=False,
                       order=1,
                       intra=False,
                       length=1.5)
                  for c in covale if c[0][0]!=c[1][0]]

        # make sure covale atoms exist
        covale = [c for c in covale if \
                  c.a[0] in return_chains.keys() and \
                  c.b[0] in return_chains.keys() and \
                  c.a in return_chains[c.a[0]].atoms.keys() and \
                  c.b in return_chains[c.b[0]].atoms.keys()]

        
        # fix charges and hydrogen counts for cases when
        # charged a atom is connected by an inter-residue bond
        for bond in bonds+covale:
            if bond.intra==False:
                for i in (bond.a,bond.b):
                    a = return_chains[i[0]].atoms[i]
                    
                    #if a.element==7 and a.charge==1 and a.hyb==3 and a.nhyd==3 and a.hvydeg==1: # -NH3+
                    #    return_chains[i[0]].atoms[i] = a._replace(charge=0, hyb=2, nhyd=1)
                    if a.element==7 and a.charge==1 and a.hyb==3 and a.nhyd==2 and a.hvydeg==2: # -(NH2+)-
                        return_chains[i[0]].atoms[i] = a._replace(charge=0, hyb=2, nhyd=0)
                    elif a.element==7 and a.charge==1 and a.hyb==3 and a.nhyd==3 and a.hvydeg==0: # free NH3+ group
                        return_chains[i[0]].atoms[i] = a._replace(charge=0, hyb=2, nhyd=2)
                    elif a.element==8 and a.charge==-1 and a.hyb==3 and a.nhyd==0:
                        return_chains[i[0]].atoms[i] = a._replace(charge=0)
                    elif a.element==8 and a.charge==-1 and a.hyb==2 and a.nhyd==0: # O-linked connections
                        return_chains[i[0]].atoms[i] = a._replace(charge=0)
                    elif a.charge!=0:
                        #print(filename, i, a)
                        pass

                '''
                a = return_chains[bond.a[0]].atoms[bond.a]
                b = return_chains[bond.b[0]].atoms[bond.b]

                # nitrogen (involved in a peptide bond)
                if a.element==7 and a.charge==1 and a.hyb==3 and (a.nhyd==3 or (a.nhyd==2 and bond.a[2]=='PRO')):
                    return_chains[bond.a[0]].atoms[bond.a] = a._replace(charge=0, hyb=2, nhyd=1)
                if b.element==7 and b.charge==1 and b.hyb==3 and (b.nhyd==3 or (b.nhyd==2 and bond.b[2]=='PRO')):
                    return_chains[bond.b[0]].atoms[bond.b] = b._replace(charge=0, hyb=2, nhyd=1)
            
                # oxygen (nucletides)
                if a.element==8 and a.charge==-1 and a.hyb==3 and a.nhyd==0:
                    return_chains[bond.a[0]].atoms[bond.a] = a._replace(charge=0)
                if b.element==8 and b.charge==-1 and b.hyb==3 and b.nhyd==0:
                    return_chains[bond.b[0]].atoms[bond.b] = b._replace(charge=0)
                    
                a = return_chains[bond.a[0]].atoms[bond.a]
                b = return_chains[bond.b[0]].atoms[bond.b]
                if a.charge!=0:
                    print(filename, bond.a,bond.b, a)
                if b.charge!=0:
                    print(filename, bond.b,bond.a, b)
                '''

        res = None
        if data.getObj('refine') is not None:
            try:
                res = float(data.getObj('refine').getValue('ls_d_res_high',0))
            except:
                res = None
        if (data.getObj('em_3d_reconstruction') is not None) and (res is None):
            try:
                res = float(data.getObj('em_3d_reconstruction').getValue('resolution',0))
            except:
                res = None

        meta = {
            'method' : data.getObj('exptl').getValue('method',0).replace(' ','_'),
            'date' : data.getObj('pdbx_database_status').getValue('recvd_initial_deposition_date',0),
            'resolution' : res
        }

        return return_chains,asmb,covale,meta
    
    
    #@staticmethod
    def save(self, chain : Chain, filename : str):
        '''save a single chain'''
        
        with open(filename, 'w') as f:
            acount = 1
            a2i = {}
            for r,a in chain.atoms.items():
                if a.occ>0:
                    element = self.i2a[a.element] if a.element in self.i2a.keys() else 'X'
                    f.write ("%-6s%5s %-4s %3s%2s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n"%(
                        "HETATM" if a.hetero==True else "ATOM",
                        acount, ' '*a.align+a.name[2], r[2], r[0], int(r[1]),
                        a.xyz[0], a.xyz[1], a.xyz[2], a.occ, 0.0, element, a.charge) )
                    a2i[r] = acount
                    acount += 1
            for bond in chain.bonds:
                if chain.atoms[bond.a].occ==0.0:
                    continue
                if chain.atoms[bond.b].occ==0.0:
                    continue
                if chain.atoms[bond.a].hetero==False and chain.atoms[bond.b].hetero==False:
                    continue
                f.write ("%-6s%5d%5d\n"%("CONECT", a2i[bond.a], a2i[bond.b]))


    #@staticmethod
    def save_all(self,
                 chains : Dict[str,Chain],
                 covale : List[Bond],
                 filename : str):
        '''save multiple chains'''

        #'''
        with open(filename, 'w') as f:
            acount = 1
            a2i = {}
            for chain_id,chain in chains.items():
                for r,a in chain.atoms.items():
                    if a.occ>0:
                        element = self.i2a[a.element] if a.element in self.i2a.keys() else 'X'
                        f.write ("%-6s%5s %-4s %3s%2s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n"%(
                            "HETATM" if a.hetero==True else "ATOM",
                            acount, ' '*a.align+a.name[2], r[2], chain_id, int(r[1]),
                            a.xyz[0], a.xyz[1], a.xyz[2], a.occ, 0.0, element, a.charge) )
                        a2i[r] = acount
                        acount += 1
                for bond in chain.bonds:
                    a = chain.atoms[bond.a]
                    b = chain.atoms[bond.b]
                    if a.occ==0.0 or b.occ==0.0 or (a.hetero==False and b.hetero==False):
                        continue
                    f.write ("%-6s%5d%5d\n"%("CONECT", a2i[bond.a], a2i[bond.b]))
                f.write('TER\n')
            
            for bond in covale:
                a = chains[bond.a[0]].atoms[bond.a]
                b = chains[bond.b[0]].atoms[bond.b]
                if a.occ==0.0 or b.occ==0.0:
                    continue
                f.write ("%-6s%5d%5d\n"%("CONECT", a2i[bond.a], a2i[bond.b]))
