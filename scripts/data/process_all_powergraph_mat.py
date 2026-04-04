#!/usr/bin/env python3
"""
Process all PowerGraph .mat files using h5py (v7.3 support) and write per-file JSON metadata.
Saves to data/processed/powergraph_graphs/*.json
"""
from pathlib import Path
import json
import os
from scipy import io as sio
import h5py

DATA_ROOT = Path('data')
MAT_DIR = DATA_ROOT / 'powergraph' / 'dataset_cascades_extracted' / 'dataset_cascades'
OUT_DIR = DATA_ROOT / 'processed' / 'powergraph_graphs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

mat_files = list(MAT_DIR.rglob('*.mat'))
print(f'Found {len(mat_files)} .mat files')

for f in mat_files:
    # create unique name based on relative path
    rel = f.relative_to(MAT_DIR)
    name = str(rel).replace(os.sep, '__')
    out_meta = OUT_DIR / f'{name}.json'
    meta = {'source': str(f.relative_to(DATA_ROOT)), 'vars': {}}
    try:
        # prefer scipy.loadmat for v7 or older
        try:
            data = sio.loadmat(str(f), squeeze_me=True, struct_as_record=False)
            keys = [k for k in data.keys() if not k.startswith('__')]
            for k in keys:
                try:
                    v = data.get(k)
                    if hasattr(v, 'shape'):
                        meta['vars'][k] = {'shape': tuple(v.shape), 'dtype': str(v.dtype)}
                    else:
                        meta['vars'][k] = {'type': type(v).__name__}
                except Exception:
                    meta['vars'][k] = {'type': 'unknown'}
        except NotImplementedError:
            # v7.3 -> use h5py
            with h5py.File(str(f),'r') as hf:
                for k in hf.keys():
                    try:
                        arr = hf[k][()]
                        meta['vars'][k] = {'shape': tuple(arr.shape), 'dtype': str(arr.dtype)}
                    except Exception:
                        meta['vars'][k] = {'type': 'h5obj'}
        # write metadata
        with open(out_meta,'w') as wf:
            json.dump(meta, wf, indent=2)
        print(f'Wrote {out_meta.name} (vars: {len(meta["vars"])})')
    except Exception as e:
        print(f'Error processing {f.name}: {e}')

print('Done')
