#Temporary dataloading for PDBs. Cannot handle ligands. Lacks feature parity
#with fused_mpnn.

import collections
import itertools

import numpy as np
import prody
prody.confProDy(verbosity='none')

Atom = collections.namedtuple('Atom', [
    'name',
    'xyz', # Cartesian coordinates of the atom
    'occ', # occupancy
    'bfac' # B-factor
])

Chain = collections.namedtuple('Chain', [
    'id',
    'type',
    'atoms',
    'sequence'
])

class PDBParser(object):
    def __init__(self):
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

        self.restype_3_to_1_not_polymer_unique = {
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
            'DA': 'A',
            'DC': 'C',
            'DG': 'G',
            'DT': 'T',
            'DX': 'X',
            'A': 'A',
            'C': 'C',
            'G': 'G',
            'U': 'U',
            'RX': 'X',
            'MAS': '-',
            'PAD': '+'
        }


    def parse(self, filename):
        atoms = prody.parsePDB(filename)

        chain_letters = list(atoms.getChids())
        resnums = list(atoms.getResnums())
        resnames = list(atoms.getResnames())
        atom_names = list(atoms.getNames())
        xyzs = list(atoms.getCoords())
        occs = list(atoms.getOccupancies())
        bfacs = list(atoms.getBetas())

        atoms = []
        for (chain_letter, resnum, resname, atom_name, xyz, occ, bfac) in \
            zip(chain_letters, resnums, resnames, atom_names, xyzs, occs, bfacs):
            name = (chain_letter, str(resnum), resname, atom_name)
            atom = Atom(name=name,
                        xyz=list(xyz),
                        occ=occ,
                        bfac=bfac
                        )
            atoms.append(atom)

        atoms_by_chain = itertools.groupby(atoms, lambda x: x.name[0])
        atoms_by_chain = list(map(lambda x: (x[0], list(x[1])), atoms_by_chain))
        atoms_by_chain = dict(atoms_by_chain)

        chains = dict()
        for chain_letter in atoms_by_chain:
            chain_atoms = atoms_by_chain[chain_letter]

            unique_residues = list(set([a.name[2] for a in chain_atoms]))

            chain_is_protein = False
            chain_is_dna = False
            chain_is_rna = False
            for unique_residue in unique_residues:
                if unique_residue in self.protein_restypes:
                    chain_is_protein = True
                elif unique_residue in self.dna_restypes:
                    chain_is_dna = True
                elif unique_residue in self.rna_restypes:
                    chain_is_rna = True
            
            if (chain_is_protein) and (not chain_is_dna) and (not chain_is_rna):
                chain_type = "polypeptide(L)"
            elif (not chain_is_protein) and (chain_is_dna) and (not chain_is_rna):
                chain_type = "polydeoxyribonucleotide"
            elif (not chain_is_protein) and (not chain_is_dna) and (chain_is_rna):
                chain_type = "polyribonucleotide"
            elif (not chain_is_protein) and (chain_is_dna) and (chain_is_rna):
                chain_type = "polydeoxyribonucleotide/polyribonucleotide hybrid"
            else:
                raise Exception("Chain has a combination of residue types not supported.")

            chain_atoms_names = list(map(lambda atom: atom.name, chain_atoms))
            chain_L = len(list(set([a[1] for a in chain_atoms_names])))
            raw_sequence = chain_L * ["UNK"]
            for c, (res_id, res_atoms) in enumerate(itertools.groupby(chain_atoms_names, lambda x: x[1])):
                for atom_key in res_atoms:
                    _, res_idx_str, res_name, atom_name = atom_key
                    raw_sequence[c] = res_name
            raw_sequence = "".join(list(map(lambda resname: self.restype_3_to_1_not_polymer_unique[resname], raw_sequence)))
            
            chain_atoms_dict = dict(list(map(lambda x: (x.name, x), chain_atoms)))
            
            chain = Chain(id=chain_letter,
                          type=chain_type,
                          atoms=chain_atoms_dict,
                          sequence=raw_sequence)
            
            chains[chain_letter] = chain

        # Create the default assembly dictionary (identity matrix), so no
        # rotation or translation.
        asmb = dict()
        asmb["1"] = []
        for chain_letter in chains.keys():
            asmb["1"].append((chain_letter, np.eye(4)))
        
        covalei = None
        meta = None
        
        return chains, asmb, covalei, meta