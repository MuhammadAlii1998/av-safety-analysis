import pandas as pd

def extract_nuscenes_features(nuscenes):
    features = []
    for scene in nuscenes.scene[:5]:  # Demo: First five scenes
        name = scene['name']
        nbr_samples = scene['nbr_samples']
        features.append({'scene_name': name, 'nbr_samples': nbr_samples})
    return pd.DataFrame(features)
