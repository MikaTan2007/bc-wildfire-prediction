import torch
import torch.nn as nn
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

# uvicorn main:app
# http://127.0.0.1:8000/docs

# BC Bounding Box
BC_MIN_LAT = 48.2
BC_MAX_LAT = 60.0
BC_MIN_LON = -139.1
BC_MAX_LON = -114.0
GRID_ROWS = 25
GRID_COLS = 40

# Neural Network Architecture
class WildfireClassifier(nn.Module):
    def __init__(self, input_dim):
        super(WildfireClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1) # We output to a single node for binary classification
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
class ForecastInput(BaseModel):
    precipitation: float
    temperature: float
    dewpoint: float

class GridPoint(BaseModel):
    lat: float
    lon: float

class GridCell(BaseModel):
    grid_id: str
    row: int
    col: int
    centroid: GridPoint
    polygon: List[GridPoint]

class GridResponse(BaseModel):
    generated_at: str
    cell_count: int
    rows: int
    cols: int
    cells: List[GridCell]

def generate_bc_grid(rows: int = GRID_ROWS, cols: int = GRID_COLS) -> List[GridCell]:
    lat_step = (BC_MAX_LAT - BC_MIN_LAT) / rows
    lon_step = (BC_MAX_LON - BC_MIN_LON) / cols
    cells: List[GridCell] = []

    for r in range(rows):
        for c in range(cols):
            cell_min_lat = BC_MIN_LAT + (r * lat_step)
            cell_max_lat = cell_min_lat + lat_step
            cell_min_lon = BC_MIN_LON + (c * lon_step)
            cell_max_lon = cell_min_lon + lon_step

            centroid = GridPoint(
                lat = (cell_min_lat + cell_max_lat) / 2.0,
                lon = (cell_min_lon + cell_max_lon) / 2.0
            )

            polygon = [
                GridPoint(lat = cell_min_lat, lon = cell_min_lon),
                GridPoint(lat = cell_min_lat, lon = cell_max_lon),
                GridPoint(lat = cell_max_lat, lon = cell_min_lon),
                GridPoint(lat = cell_max_lat, lon = cell_max_lon),
                GridPoint(lat = cell_min_lat, lon = cell_min_lon)
            ]

            cells.append(
                GridCell(
                    grid_id = f"bc-r{r:02d}-c{c:02d}",
                    row = r,
                    col = c,
                    centroid = centroid,
                    polygon = polygon
                )
            )
    
    return cells

# Defining Lifespan Context Manager
ml_assets = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Model and Scaler")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = WildfireClassifier(input_dim=3)
    model.load_state_dict(torch.load("wildfire_model.pth", map_location=device))

    model.to(device)
    model.eval()

    scaler = joblib.load("scaler.joblib")

    # Store in a dictionary so they are accessible by routes
    ml_assets["model"] = model
    ml_assets["scaler"] = scaler
    ml_assets["device"] = device

    yield 

    print("Cleaning up ML Assets")
    ml_assets.clear()

    
app = FastAPI(
    title = "BC Wildfire 72-Hour Prediction API",
    version = "1.0.0",
    lifespan = lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# Routes

@app.get("/")
async def root():
    return {
        "message": "BC Wildfire Prediction API is Online", 
        "device": str(ml_assets.get("device"))
    }

@app.get("/grid", response_model = GridResponse)
async def get_grid():
    cells = generate_bc_grid(rows = GRID_ROWS, cols = GRID_COLS)
    return GridResponse(
        generated_at = datetime.now(timezone.utc).isoformat(),
        cell_count = len(cells),
        rows = GRID_ROWS,
        cols = GRID_COLS,
        cells = cells
    )

@app.post("/predict")
async def predict_wildfire(data: ForecastInput):
    model = ml_assets["model"]
    scaler = ml_assets["scaler"]
    device = ml_assets["device"]

    raw_features = np.array(
        [
            [data.precipitation, data.temperature, data.dewpoint]
        ]
    )

    scaled_features = scaler.transform(raw_features)
    input_tensor = torch.FloatTensor(scaled_features).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()
    
    return {
        "prediction_window": "72 Hours",
        "fire_probability": round(probability * 100, 2),
        "risk_level": "High" if probability > 0.75 else "Moderate" if probability > 0.4 else "Low",
        "unit": "Percentage"
    }