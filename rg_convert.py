import os

import numpy as np
import pandas as pd
import uproot as ur

BRANCH_PT   = 'TrackPt'
BRANCH_P    = 'TrackP'
BRANCH_PDG = 'MCParticleType'

branches = [
        BRANCH_PT,
        BRANCH_P,
        'NumLongTracks',
        BRANCH_PDG,
        'RichDLLbt',
        'RichDLLk',
        'RichDLLmu',
        'RichDLLp',
        'RichDLLe'
    ]

particles = {
          11 : 'electron',
          13 : 'muon',
         211 : 'pion',
         321 : 'kaon',
        2212 : 'proton',
           0 : 'ghost',
    }

def read_raw_file(fname, treename='tree'):
    root_file = ur.open(fname)
    return root_file[treename].pandas.df(branches)

def add_eta(df, branchname='TrackEta'):
    assert branchname not in df.columns
    df[branchname] = np.arccosh(df[BRANCH_P] / df[BRANCH_PT])
    return branchname

def filter_ptype(df, pdg, copy=False):
    particle = particles[pdg]
    result = df[np.abs(df[BRANCH_PDG]) == pdg]
    if copy:
        return result.copy()
    else:
        return result

def to_csv(df, prefix, particle, mode):
    fname = "{}_{}_{}.csv".format(prefix, particle, mode)
    df.to_csv(fname, index=False)

def convert(
        infile,
        outfile_prefix,
        val_test_split,
        val_size=0.2,
        seed=None
        ):
    if val_test_split:
        assert seed is not None
    raw_df = read_raw_file(infile)
    eta_branch_name = add_eta(raw_df)
    raw_df.drop(columns=BRANCH_PT, inplace=True)
    
    for pdg, particle in particles.items():
        filtered_df = filter_ptype(raw_df, pdg)
        
        if val_test_split:
            N = len(filtered_df)
            val_N = int(val_size * N)
            np.random.seed(seed)
            shuffle = np.random.permutation(N)
            val, train = shuffle[:val_N], shuffle[val_N:]
            to_csv(filtered_df.iloc[val  ], outfile_prefix, particle, "validation")
            to_csv(filtered_df.iloc[train], outfile_prefix, particle, "test"      )
        else:
            to_csv(filtered_df, outfile_prefix, particle, "train")

convert('input/global_train.root', 'output/v1', val_test_split=False)
convert('input/global_test.root' , 'output/v1', val_test_split=True , seed=42)
