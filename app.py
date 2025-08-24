from flask import Flask, request, jsonify, send_file
import rasterio
import numpy as np
import pandas as pd
import joblib
from io import BytesIO
from PIL import Image
import base64
from flask_cors import CORS

# Configuration de l'application Flask avec CORS
app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    },
    r"/download": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET"]
    }
})

model = joblib.load("random_forest_amarante.pkl")

def calculer_indices(img):
    red, green, blue, nir = [img[i].astype('float32') for i in range(4)]
    eps = 1e-6
    NDVI = (nir - red) / (nir + red + eps)
    SAVI = ((nir - red) / (nir + red + 0.5 + eps)) * 1.5
    GNDVI = (nir - green) / (nir + green + eps)
    NDWI = (green - nir) / (green + nir + eps)
    MSAVI = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
    VARI = (green - red) / (green + red - blue + eps)
    ExG = 2 * green - red - blue
    EVI = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
    return np.stack([NDVI, SAVI, GNDVI, NDWI, MSAVI, VARI, ExG, EVI], axis=-1)

def map_color(proba):
    if np.isnan(proba) or proba == 1.0:
        return [255, 255, 255]
    elif proba < 0.4:
        return [0, 255, 0]
    elif proba < 0.6:
        return [255, 255, 0]
    elif proba < 0.7:
        return [255, 165, 0]
    else:
        return [255, 0, 0]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        print("[INFO] Fichier reçu :", file.filename)

        with rasterio.open(file) as src:
            img = src.read()
            profile = src.profile
            transform = src.transform

        print("[INFO] Image shape :", img.shape)
        if img.shape[0] < 4:
            raise ValueError("L'image doit avoir au moins 4 bandes (R, G, B, NIR).")

        features = calculer_indices(img)
        rows, cols = features.shape[:2]
        flat_features = features.reshape(-1, features.shape[-1])

        probas = model.predict_proba(flat_features)[:, 1]
        proba_image = probas.reshape((rows, cols))

        color_image = np.zeros((rows, cols, 3), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                color_image[i, j] = map_color(proba_image[i, j])

        image_pil = Image.fromarray(color_image)
        buffered = BytesIO()
        image_pil.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        high_prob_indices = np.argwhere(proba_image > 0.8)
        lon_lat_list = []
        for r, c in high_prob_indices:
            lon, lat = transform * (c, r)
            lon_lat_list.append({"Latitude": lat, "Longitude": lon, "Proba": float(proba_image[r, c])})

        df = pd.DataFrame(lon_lat_list)
        df.to_csv("points_probabilite_sup_08.csv", index=False)

        return jsonify({
            "heatmap": heatmap_base64,
            "points": lon_lat_list
        })

    except Exception as e:
        print("[ERREUR SERVEUR]", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/download", methods=["GET"])
def download():
    return send_file("points_probabilite_sup_08.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)