import sys
import glob
import itertools
import pandas as pd
from openbabel import openbabel

sys.path.append("/home/akubaney/projects/na_mpnn")
import cifutils

openbabel.obErrorLog.SetOutputLevel(0)

start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])

fnames = glob.glob('/databases/rcsb/cif/*/*.cif.gz')
fnames.sort()

Parser = cifutils.CIFParser(skip_res=['HOH'])

data = {'label':[],'date':[],'method':[],'resolution':[],'poly':[],'poly_type':[],'nonpoly':[],'num_heavy':[],'coverage':[],'poly_sequence':[]}

for i,fname in enumerate(fnames[start_idx:end_idx]):
    try:
        chains,asmb,covale,meta = Parser.parse(fname)
        #indices = {int(a[1]) for c in chains.values() if c.type=='nonpoly' for a in c.atoms.keys()}
        #hydrogens = [a for c in chains.values() for a in c.atoms.values() if a.element==1 and a.occ>0]
        #print(fname, len(chains), len(covale), len(indices), len(hydrogens))
        #Parser.save_all(chains,covale,'/dev/null')

        heavy_atoms = [a for c in chains.values() for a in c.atoms.values() if a.element>1]
        m,n = 0,0
        for g in itertools.groupby(heavy_atoms, key=lambda a : a.name[:3]):
                res_atoms = list(g[1])
                nobs = sum([a.occ>0 for a in res_atoms])
                m += nobs
                if nobs>0:
                    n += len(res_atoms)

        label = fname.split('/')[-1][:-7]

        data['label'].append(label)
        data['method'].append(meta['method'])
        data['resolution'].append(meta['resolution'])
        data['date'].append(meta['date'])
        data['poly'].append([k for k,v in chains.items() if 'nonpoly' not in v.type])
        data['poly_type'].append([v.type for k,v in chains.items() if 'nonpoly' not in v.type])
        data['poly_sequence'].append([v.sequence for k,v in chains.items() if 'nonpoly' not in v.type])
        data['nonpoly'].append([k for k,v in chains.items() if 'nonpoly' in v.type])
        data['num_heavy'].append(n)
        data['coverage'].append(m/n if n>0 else 0)
    except:
        print('ERROR:', fname)

    #sys.stdout.flush()
#print('done')

df = pd.DataFrame.from_dict(data)
df.to_csv('pdb_content/%d_%d.csv'%(start_idx, end_idx),index=False)
