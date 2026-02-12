import os
import numpy as np
import cv2
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


# -----------------------------
# Load scaler + TFLite once
# -----------------------------
SCALER_PATH = "feature_scaler_FINAL.pkl"
TFLITE_PATH = "spad_mlp.tflite"

scaler = joblib.load(SCALER_PATH)

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = FastAPI(title="SPAD Predictor API")

# -----------------------------
# Feature extraction (training match)
# Center-crop 224x224, then compute 25 features
# -----------------------------
FEATURE_COLUMNS = [
    'Red','Green','Blue',
    'Yellow',
    'Hue','Saturation','Value',
    'Brightness',
    'L','a*','b*',
    'Y','Cr','Cb',
    'Key',
    'ExG','NExG','ExR','ExGR',
    'GRVI','VARI','GLI',
    'TGI','CIVE','VeG'
]

def extract_features_from_bgr(img_bgr: np.ndarray) -> np.ndarray:
    # ----- center crop 224x224 -----
    h, w = img_bgr.shape[:2]
    if h < 224 or w < 224:
        raise ValueError(f"Image too small for 224x224 crop: {w}x{h}")

    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img_bgr = img_bgr[y0:y0+224, x0:x0+224]

    # ----- features -----
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]
    eps = 1e-6

    mean_R, mean_G, mean_B = float(np.mean(R)), float(np.mean(G)), float(np.mean(B))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    mean_H = float(np.mean(hsv[:, :, 0]))
    mean_S = float(np.mean(hsv[:, :, 1]))
    mean_V = float(np.mean(hsv[:, :, 2]))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_L = float(np.mean(lab[:, :, 0]))
    mean_A = float(np.mean(lab[:, :, 1]))
    mean_B_lab = float(np.mean(lab[:, :, 2]))

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    mean_Y  = float(np.mean(ycrcb[:, :, 0]))
    mean_Cr = float(np.mean(ycrcb[:, :, 1]))
    mean_Cb = float(np.mean(ycrcb[:, :, 2]))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    # CMYK (Yellow + Key)
    Rn = R / 255.0
    Gn = G / 255.0
    Bn = B / 255.0

    K = 1 - np.maximum(np.maximum(Rn, Gn), Bn)
    K_inv = 1 - K
    K_inv[K_inv == 0] = eps

    Y_cmyk = (1 - Bn - K) / K_inv
    Yellow = float(np.mean(Y_cmyk) * 100.0)
    Key = float(np.mean(K) * 100.0)

    exg  = (2*mean_G - mean_R - mean_B)
    nexg = (2*mean_G - mean_R - mean_B) / (mean_R + mean_G + mean_B + eps)
    exr  = (1.4*mean_R - mean_G)
    exgr = exg - exr
    grvi = (mean_G - mean_R) / (mean_G + mean_R + eps)
    vari = (mean_G - mean_R) / (mean_G + mean_R - mean_B + eps)
    gli  = (2*mean_G - mean_R - mean_B) / (2*mean_G + mean_R + mean_B + eps)
    tgi  = (-0.5) * ((110*(mean_R - mean_G) - 180*(mean_R - mean_B)))
    cive = (0.441*mean_R - 0.811*mean_G + 0.385*mean_B + 18.78745)
    veg  = mean_G / (pow(mean_R, 0.667) * pow(mean_B, 0.333) + eps)

    feature_dict = {
        'Red':mean_R,'Green':mean_G,'Blue':mean_B,
        'Yellow':Yellow,
        'Hue':mean_H,'Saturation':mean_S,'Value':mean_V,
        'Brightness':brightness,
        'L':mean_L,'a*':mean_A,'b*':mean_B_lab,
        'Y':mean_Y,'Cr':mean_Cr,'Cb':mean_Cb,
        'Key':Key,
        'ExG':exg,'NExG':nexg,'ExR':exr,'ExGR':exgr,
        'GRVI':grvi,'VARI':vari,'GLI':gli,
        'TGI':tgi,'CIVE':cive,'VeG':veg
    }

    feats = np.array([feature_dict[c] for c in FEATURE_COLUMNS], dtype=np.float32)
    if feats.shape != (25,):
        raise RuntimeError(f"Feature shape mismatch: {feats.shape}")
    return feats

def predict_spad_from_bgr(img_bgr: np.ndarray) -> float:
    feats = extract_features_from_bgr(img_bgr).reshape(1, 25)
    feats_scaled = scaler.transform(feats).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], feats_scaled)
    interpreter.invoke()
    spad = interpreter.get_tensor(output_details[0]["index"])[0][0]
    return float(spad)

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "SPAD API running. Use POST /predict with an image."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        np_arr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        spad = predict_spad_from_bgr(img_bgr)
        return {"spad": round(spad, 2)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

