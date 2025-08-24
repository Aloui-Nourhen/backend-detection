import rasterio
import numpy as np
import joblib
from rasterio.transform import Affine

# Chargement du modèle une fois au démarrage
model = joblib.load("random_forest_amarante.pkl")

def calculate_indices(img):
    red = img[0].astype('float32')
    green = img[1].astype('float32')
    blue = img[2].astype('float32')
    nir = img[3].astype('float32')
    epsilon = 1e-6

    NDVI = (nir - red) / (nir + red + epsilon)
    SAVI = ((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5
    GNDVI = (nir - green) / (nir + green + epsilon)
    NDWI = (green - nir) / (green + nir + epsilon)
    MSAVI = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
    VARI = (green - red) / (green + red - blue + epsilon)
    ExG = 2 * green - red - blue
    EVI = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + epsilon)

    return np.stack([NDVI, SAVI, GNDVI, NDWI, MSAVI, VARI, ExG, EVI], axis=-1)

def predict_amarante(tif_path):
    with rasterio.open(tif_path) as src:
        img = src.read()
        transform = src.transform
        bounds = src.bounds

    features, shape = calculate_indices(img)
    flat_features = features.reshape(-1, features.shape[-1])

    probas = model.predict_proba(flat_features)[:, 1]
    proba_image = probas.reshape(shape)

    # Détection des zones infectées
    high_prob_indices = np.argwhere(proba_image > 0.8)
    infected_areas = []
    
    for r, c in high_prob_indices:
        lon, lat = transform * (c, r)
        infected_areas.append({
            "lat": lat,
            "lon": lon,
            "probability": float(proba_image[r, c])
        })

    return {
        "heatmap": proba_image,
        "infected_areas": infected_areas,
        "bounds": {
            "sw": [bounds.bottom, bounds.left],
            "ne": [bounds.top, bounds.right]
        }
    }