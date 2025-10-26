import pandas as pd
from nuscenes.nuscenes import NuScenes

def load_nhtsa_data(path):
    return pd.read_csv(path)

def load_nuscenes_data(dataroot):
    return NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
