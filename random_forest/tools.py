"""
https://www.kaggle.com/gregbehm/plasticc-data-reader-get-objects-by-id 
"""

from collections import deque
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def get_objects_by_id(path, chunksize=1_000_000):
    """
    Generator that iterates over chunks of PLAsTiCC Astronomical Classification challenge
    data contained in the CVS file at path.
    
    Yields subsequent (object_id, pd.DataFrame) tuples, where each DataFrame contains
    all observations for the associated object_id.
    
    Inputs:
        path: CSV file path name
        chunksize: iteration chunk size in rows
        
    Output:
        Generator that yields (object_id, pd.DataFrame) tuples
    """
    
    # set initial state
    last_id = None
    last_df = pd.DataFrame()
    
    for df in pd.read_csv(path, chunksize=chunksize):
        
        # Group by object_id; store grouped dataframes into dict for fast access
        grouper = {
            object_id: pd.DataFrame(group)
            for object_id, group in df.groupby('object_id')
        }

        # queue unique object_ids, in order, for processing
        object_ids = df['object_id'].unique()
        queue = deque(object_ids)

        # if the object carried over from previous chunk matches
        # the first object in this chunk, stitch them together
        first_id = queue[0]
        if first_id == last_id:
            first_df = grouper[first_id]
            last_df = pd.concat([last_df, first_df])
            grouper[first_id] = last_df
        elif last_id is not None:
            # save last_df and return as first result
            grouper[last_id] = last_df
            queue.appendleft(last_id)
        
        # save last object in chunk
        last_id = queue[-1]
        last_df = grouper[last_id]

        # check for edge case with only one object_id in this chunk
        if first_id == last_id:
            # yield nothing for now...
            continue
        
        # yield all but last object, which may be incomplete in this chunk
        while len(queue) > 1:
            object_id = queue.popleft()
            object_df = grouper.pop(object_id)
            yield (object_id, object_df)
    
    # yield remaining object
    yield (last_id, last_df)