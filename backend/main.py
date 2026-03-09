# from cProfile import label

# from fastapi import FastAPI, File, UploadFile
# import torch
# from torchvision import transforms, models
# from PIL import Image
# import io

# app = FastAPI()

# MODEL_PATH = "D:/PlantData/saved_model/plant_disease_model.pth"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint = torch.load(MODEL_PATH, map_location=device)
# class_names = checkpoint["class_names"]

# model = models.efficientnet_b0(weights=None)
# model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
# model.load_state_dict(checkpoint["model_state_dict"])
# model.to(device)
# model.eval()

# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],
#                          [0.229,0.224,0.225])
# ])

# @app.post("/predict")
# async def predict(file: UploadFile =File(...)):
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).convert("RGB")
#     image = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = model(image)
#         _, pred = torch.max(output, 1)

#     # return {"prediction": class_names[pred.item()]}
#     label = class_names[pred.item()]

#     parts = label.split("___")
#     crop = parts[0]
#     disease = parts[1]

#     status = "Healthy" if "healthy" in disease.lower() else "Diseased"

#     # return {
#     # "crop": crop,
#     # "status": status,
#     # "disease": disease
#     # }

#     import torch.nn.functional as F

#     label = class_names[pred.item()]
#     confidence = torch.max(F.softmax(output, dim=1)).item()

#     parts = label.split("___")
#     crop = parts[0]
#     disease = parts[1]
#     status = "Healthy" if "healthy" in disease.lower() else "Diseased"

#     return {
#     "crop": crop,
#     "status": status,
#     "disease": disease,
#     "confidence": round(confidence * 100, 2)
# }


# # backend/main.py
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from PIL import Image
# import io
# import torch
# import torch.nn.functional as F

# from .model import load_model, get_transform  # relative import since it's in backend/

# app = FastAPI(title="Plant Disease API", version="1.0")

# # Allow your Flutter client; during dev, * is fine. In prod, set your domains.
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],            # TODO: restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Use relative path so it works on Render
# MODEL_PATH = "saved_model/plant_disease_model.pth"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @app.on_event("startup")
# async def startup_event():
#     # Load model and keep references on app.state
#     app.state.model, app.state.class_names = load_model(MODEL_PATH, device)
#     app.state.transform = get_transform()

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid image file")

#     image_t = app.state.transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = app.state.model(image_t)
#         _, pred = torch.max(output, 1)
#         confidence = torch.max(F.softmax(output, dim=1)).item()

#     label = app.state.class_names[pred.item()]
#     parts = label.split("___")
#     crop = parts[0]
#     disease = parts[1] if len(parts) > 1 else label
#     status = "Healthy" if "healthy" in disease.lower() else "Diseased"

#     return JSONResponse({
#         "crop": crop,
#         "status": status,
#         "disease": disease,
#         "confidence": round(confidence * 100, 2)
#     })

# # backend/main.py
# from fastapi import FastAPI, File, UploadFile, HTTPException, Depends  # NEW: Depends
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi.security import APIKeyHeader  # NEW
# from PIL import Image
# import os  # NEW
# import io
# import torch
# import torch.nn.functional as F

# from .model import load_model, get_transform  # relative import since it's in backend/

# app = FastAPI(title="Plant Disease API", version="1.1")

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # TODO: restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- API Key security (configurable via env) ---  # NEW
# API_KEY_HEADER_NAME = os.getenv("API_KEY_HEADER_NAME", "x-api-key")
# REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "true").lower() == "true"
# EXPECTED_API_KEY = os.getenv("API_KEY")  # set in Render dashboard
# api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

# async def verify_api_key(x_api_key: str = Depends(api_key_header)):
#     if not REQUIRE_API_KEY:
#         return
#     if not EXPECTED_API_KEY:
#         # Misconfiguration (REQUIRE_API_KEY=true but no API_KEY set)
#         raise HTTPException(status_code=500, detail="Server security not configured")
#     if x_api_key != EXPECTED_API_KEY:
#         raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")

# # Use relative path so it works on Render
# MODEL_PATH = os.getenv("MODEL_PATH", "saved_model/plant_disease_model.pth")  # NEW
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @app.on_event("startup")
# async def startup_event():
#     app.state.model, app.state.class_names = load_model(MODEL_PATH, device)
#     app.state.transform = get_transform()

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...),
#     _auth: None = Depends(verify_api_key),  # NEW: protect with API key
# ):
#     # Optional: basic content-type validation
#     if not (file.content_type or "").lower().startswith("image/"):  # NEW
#         raise HTTPException(status_code=400, detail="Please upload an image file")

#     try:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid image file")

#     image_t = app.state.transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = app.state.model(image_t)
#         _, pred = torch.max(output, 1)
#         confidence = torch.max(F.softmax(output, dim=1)).item()

#     label = app.state.class_names[pred.item()]
#     parts = label.split("___")
#     crop = parts[0]
#     disease = parts[1] if len(parts) > 1 else label
#     status = "Healthy" if "healthy" in disease.lower() else "Diseased"

#     message = f"crop: {crop}\nstatus: {status}\ndisease: {disease}"  # optional farmer-friendly message

#     return JSONResponse({
#         "crop": crop,
#         "status": status,
#         "disease": disease,
#         "confidence": round(confidence * 100, 2),
#         "message": message  # optional
#     })



# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from PIL import Image
import os
import io
import torch
import torch.nn.functional as F

from .model import load_model, get_transform

app = FastAPI(title="Plant Disease API", version="1.1")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # TODO: restrict to your app domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Key security (configurable via env) ---
API_KEY_HEADER_NAME = os.getenv("API_KEY_HEADER_NAME", "x-api-key")
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "true").lower() == "true"
EXPECTED_API_KEY = os.getenv("API_KEY")  # set this in Render -> Environment
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

async def verify_api_key(x_api_key: str = Depends(api_key_header)):
    if not REQUIRE_API_KEY:
        return
    if not EXPECTED_API_KEY:
        # Misconfiguration (REQUIRE_API_KEY=true but no API_KEY set)
        raise HTTPException(status_code=500, detail="Server security not configured")
    if x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")

# Use relative path so it works on Render
MODEL_PATH = os.getenv("MODEL_PATH", "saved_model/plant_disease_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def startup_event():
    app.state.model, app.state.class_names = load_model(MODEL_PATH, device)
    app.state.transform = get_transform()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    _auth: None = Depends(verify_api_key),
):
    # Validate content-type
    if not (file.content_type or "").lower().startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    # Read and parse image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Preprocess & infer
    image_t = app.state.transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = app.state.model(image_t)
        _, pred = torch.max(output, 1)
        confidence = torch.max(F.softmax(output, dim=1)).item()

    # Post-process label
    label = app.state.class_names[pred.item()]
    parts = label.split("___")
    crop = parts[0]
    disease = parts[1] if len(parts) > 1 else label
    status = "Healthy" if "healthy" in disease.lower() else "Diseased"

    message = f"crop: {crop}\nstatus: {status}\ndisease: {disease}"

    return JSONResponse({
        "crop": crop,
        "status": status,
        "disease": disease,
        "confidence": round(confidence * 100, 2),
        "message": message
    })