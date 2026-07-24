import os
import json
import gzip
import copy
from collections import defaultdict
import shutil

def process(path):
    print(f"Fixing dataset: {path} ...")
    # 1. Backup the original file (safety first)
    shutil.copy(path, path.replace('.json.gz', '-old.json.gz'))
    
    # 2. Read and decompress
    with gzip.open(path, 'rt') as f:
        ds = json.load(f)

    ds_new = copy.deepcopy(ds)
    removed_count = 0

    # 3. Iterate through each episode
    for i, e in enumerate(ds['episodes']):
        objs = []
        obj2num = defaultdict(int)
        
        # Count objects that actually exist in the scene
        for obj in e['rigid_objs']:
            obj_name = obj[0].split('.')[0]
            objs.append(obj_name + f"_:{str(obj2num[obj_name]).zfill(4)}")
            obj2num[obj_name] += 1
            
        # Check the task manifest, remove non-existent "ghost objects"
        for obj in list(e['name_to_receptacle'].keys()):
            if obj not in objs:
                del ds_new['episodes'][i]['name_to_receptacle'][obj]
                removed_count += 1

    # 4. Repack and save
    with gzip.open(path, 'wt') as f:
        json.dump(ds_new, f)
    print(f"Fix complete! Removed {removed_count} incorrect ghost object references.")


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
path_val = os.path.join(_REPO_ROOT, 'data/versioned_data/hab3-episodes/val/social_rearrange_diverse.json.gz')
process(path_val)