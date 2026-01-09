

import itertools
import pandas
import numpy as np


def isin_pair_elements(elements, test_elements):
    """ """
    elements_combined = (elements[:, 0] << 16) | elements[:, 1]
    test_elements_combined = (test_elements[:, 0] << 16) | test_elements[:, 1]
    return np.isin(elements_combined, test_elements_combined)

# pandas concat tricks suggested by: AntoineGillesLordet (https://github.com/MickaelRigault/skysurvey/issues/35)
# aranged by: Mickael Rigault
def chunk_dfs(dfs, chunk_size):
    """ """
    dfs_out = []
    for df in dfs:
        dfs_out.append(df)
        if len(dfs_out) == chunk_size:
            yield dfs_out, chunk_size
            dfs_out = []
            
    if dfs_out:
        yield dfs_out, len(dfs_out)

def concat_chunk(dfs, **kwargs):
    """ Helper to concatenate a chunk of DataFrames. """
    # Optimization: Convert generator to list immediately so pandas.concat 
    # can pre-allocate memory efficiently.
    return pandas.concat(list(dfs), **kwargs)

def eff_concat(dfs, chunk_size, keys=None, **kwargs):
    """ 
    Memory-efficient concatenation by chunking.
    Now handles keys=None and avoids itertools.tee for lists.
    """
    # Optimization: Avoid tee if input is already a list (which it is in dataset.py)
    if hasattr(dfs, '__len__'):
        total_len = len(dfs)
    else:
        dfs, dfs_len = itertools.tee(dfs, 2)
        total_len = len(list(dfs_len))

    # Fast path for small datasets
    if total_len < chunk_size:
        return concat_chunk(dfs, keys=keys, **kwargs)
    
    # Logic to handle keys=None safely during chunking
    return pandas.concat(
        (
            concat_chunk(
                chunk, 
                keys=None if keys is None else keys[i*chunk_size : i*chunk_size+l], 
                **kwargs
            )
            for i, (chunk, l) in enumerate(chunk_dfs(dfs, chunk_size))
        )
    )
