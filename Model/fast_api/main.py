import torch
import torch.nn as nn
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List
import asyncio
import httpx
import time
import json
from pathlib import Path
from shapely.geometry import Point, Polygon, shape
from shapely.prepared import prep

# uvicorn main:app
# http://127.0.0.1:8000/docs

# BC Bounding Box
BC_MIN_LAT = 48.2
BC_MAX_LAT = 60.0
BC_MIN_LON = -139.1
BC_MAX_LON = -114.0
GRID_ROWS = 25
GRID_COLS = 28

WEATHER_BATCH_SIZE = 25
WEATHER_MAX_RETRIES = 6
WEATHER_BASE_BACKOFF_SEC = 1
WEATHER_BATCH_PAUSE_SEC = 0.75

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

# Grid Generation

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

# Grid Weather Response

class GridWeather(BaseModel):
    grid_id: str
    centroid: GridPoint
    polygon: List[GridPoint]
    precipitation: float
    temperature: float
    dewpoint: float

class GridWeatherResponse(BaseModel):
    generated_at: str
    cell_count: int
    weather_source: str
    cells: List[GridWeather]

class GridPrediction(BaseModel):
    grid_id: str
    centroid: GridPoint
    polygon: List[GridPoint]
    precipitation: float
    temperature: float
    dewpoint: float
    fire_probability: float
    risk_level: str

class GridPredictionResponse(BaseModel):
    generated_at: str
    cell_count: int
    prediction_window: str
    weather_source: str
    cells: List[GridPrediction]


GRID_PREDICT_CACHE_TTL_SEC = 600  # 10 minutes
grid_predict_cache = {
    "generated_at_epoch": 0.0,
    "payload": None,
}

def generate_bc_grid(
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    min_land_overlap: float = 0.15,  # keep cell if >=15% overlaps BC land
) -> List[GridCell]:
    lat_step = (BC_MAX_LAT - BC_MIN_LAT) / rows
    lon_step = (BC_MAX_LON - BC_MIN_LON) / cols
    cells: List[GridCell] = []

    bc_geom = ml_assets.get("bc_land_geom")

    for r in range(rows):
        for c in range(cols):
            cell_min_lat = BC_MIN_LAT + (r * lat_step)
            cell_max_lat = cell_min_lat + lat_step
            cell_min_lon = BC_MIN_LON + (c * lon_step)
            cell_max_lon = cell_min_lon + lon_step

            centroid = GridPoint(
                lat=(cell_min_lat + cell_max_lat) / 2.0,
                lon=(cell_min_lon + cell_max_lon) / 2.0,
            )

            polygon = [
                GridPoint(lat=cell_min_lat, lon=cell_min_lon),
                GridPoint(lat=cell_min_lat, lon=cell_max_lon),
                GridPoint(lat=cell_max_lat, lon=cell_max_lon),
                GridPoint(lat=cell_max_lat, lon=cell_min_lon),
                GridPoint(lat=cell_min_lat, lon=cell_min_lon),
            ]

            # Land filtering
            if bc_geom is not None:
                cell_poly = Polygon([
                    (cell_min_lon, cell_min_lat),
                    (cell_max_lon, cell_min_lat),
                    (cell_max_lon, cell_max_lat),
                    (cell_min_lon, cell_max_lat),
                ])
                overlap_ratio = cell_poly.intersection(bc_geom).area / cell_poly.area
                if overlap_ratio < min_land_overlap:
                    continue

            cells.append(
                GridCell(
                    grid_id=f"bc-r{r:02d}-c{c:02d}",
                    row=r,
                    col=c,
                    centroid=centroid,
                    polygon=polygon,
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

# Grid Generation

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

# Fetching Weather Per Point

async def fetch_weather_for_batch(client: httpx.AsyncClient, batch: List[GridCell]) -> List[GridWeather]:
    lat_csv = ",".join(str(cell.centroid.lat) for cell in batch)
    lon_csv = ",".join(str(cell.centroid.lon) for cell in batch)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat_csv,
        "longitude": lon_csv,
        "daily": ["temperature_2m_max", "precipitation_sum"],
        "hourly": ["dew_point_2m"],
        "forecast_days": 1,
        "timezone": "UTC",
        "cell_selection": "nearest",
    }

    last_error_text = ""

    for attempt in range(WEATHER_MAX_RETRIES):
        resp = await client.get(url, params=params)

        if resp.status_code == 429:
            wait_sec = WEATHER_BASE_BACKOFF_SEC * (2 ** attempt)
            await asyncio.sleep(wait_sec)
            last_error_text = resp.text
            continue

        if resp.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Open-Meteo upstream error {resp.status_code}: {resp.text}",
            )

        payload = resp.json()

        # Open-Meteo can return dict for single location, list for multiple
        payload_list = [payload] if isinstance(payload, dict) else payload

        if len(payload_list) != len(batch):
            raise HTTPException(
                status_code=502,
                detail=f"Open-Meteo returned {len(payload_list)} locations for batch size {len(batch)}",
            )

        out: List[GridWeather] = []
        for cell, wx in zip(batch, payload_list):
            daily = wx.get("daily", {})
            hourly = wx.get("hourly", {})
            precip_series = daily.get("precipitation_sum", [])
            temp_series = daily.get("temperature_2m_max", [])
            dew_series_raw = hourly.get("dew_point_2m", [])
            dew_series = [d for d in dew_series_raw if d is not None]

            if not dew_series:
                raise HTTPException(
                    status_code=502,
                    detail=f"Missing dew_point_2m for grid {cell.grid_id}",
                )

            out.append(
                GridWeather(
                    grid_id=cell.grid_id,
                    centroid=cell.centroid,
                    polygon=cell.polygon,
                    precipitation=_first_valid_float(precip_series, "precipitation_sum", cell.grid_id),
                    temperature=_first_valid_float(temp_series, "temperature_2m_max", cell.grid_id),
                    dewpoint=float(sum(dew_series) / len(dew_series)),
                )
            )

        return out

    raise HTTPException(
        status_code=503,
        detail=f"Open-Meteo rate limited after retries: {last_error_text}",
    )

def chunk_cells(cells: List[GridCell], size: int) -> List[List[GridCell]]:
    return [cells[i:i + size] for i in range(0, len(cells), size)]


def _first_valid_float(series, field_name: str, grid_id: str) -> float:
    if not series or series[0] is None:
        raise HTTPException(
            status_code=502,
            detail=f"Missing {field_name} for grid {grid_id}",
        )
    return float(series[0])

@app.get("/grid-weather", response_model=GridWeatherResponse)
async def get_grid_weather():
    cells = generate_bc_grid(rows=GRID_ROWS, cols=GRID_COLS)

    timeout = httpx.Timeout(30.0)
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=10)

    enriched: List[GridWeather] = []

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        batches = chunk_cells(cells, WEATHER_BATCH_SIZE)

        for batch in batches:
            batch_weather = await fetch_weather_for_batch(client, batch)
            enriched.extend(batch_weather)
            await asyncio.sleep(WEATHER_BATCH_PAUSE_SEC)

    return GridWeatherResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        cell_count=len(enriched),
        weather_source="open-meteo",
        cells=enriched,
    )

# Add this helper function (for shared batch inference)

def run_batch_inference(weather_rows: List[GridWeather]) -> List[GridPrediction]:
    model = ml_assets["model"]
    scaler = ml_assets["scaler"]
    device = ml_assets["device"]

    raw_features = np.array(
        [[w.precipitation, w.temperature, w.dewpoint] for w in weather_rows],
        dtype=np.float32
    )

    scaled_features = scaler.transform(raw_features)
    input_tensor = torch.FloatTensor(scaled_features).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy().reshape(-1)

    output: List[GridPrediction] = []
    for w, prob in zip(weather_rows, probabilities):
        prob -= 0.20
        if prob < 0:
            prob = 0
        risk = "High" if prob > 0.75 else "Moderate" if prob > 0.4 else "Low"
        output.append(
            GridPrediction(
                grid_id=w.grid_id,
                centroid=w.centroid,
                polygon=w.polygon,
                precipitation=w.precipitation,
                temperature=w.temperature,
                dewpoint=w.dewpoint,
                fire_probability=round(float(prob) * 100, 2),
                risk_level=risk,
            )
        )

    return output

@app.get("/grid-predict", response_model=GridPredictionResponse)
async def get_grid_predict():
    now = time.time()
    cache_age = now - float(grid_predict_cache["generated_at_epoch"])
    if grid_predict_cache["payload"] is not None and cache_age < GRID_PREDICT_CACHE_TTL_SEC:
        return grid_predict_cache["payload"]

    weather_payload = await get_grid_weather()
    preds = run_batch_inference(weather_payload.cells)

    payload = GridPredictionResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        cell_count=len(preds),
        prediction_window="72 Hours",
        weather_source=weather_payload.weather_source,
        cells=preds,
    )

    grid_predict_cache["generated_at_epoch"] = now
    grid_predict_cache["payload"] = payload
    return payload

# Architecture
"""
GET /grid-predict 
-> calls get_grid_weather 
    -> generate_bc_grid 
        -> generates the geometry for each grid
        -> chunks cells 
    -> fetches weather per batch
    > returns weather per cell
    -> feeds each cell into the prediction model
        -> returns weather + probability + risk level per cell
"""

# For testing
@app.post("/predict-batch")
async def predict_batch(data: List[ForecastInput]):
    if not data:
        raise HTTPException(status_code=400, detail="Input list is empty")

    rows = [
        GridWeather(
            grid_id=f"manual-{i}",
            centroid=GridPoint(lat=0.0, lon=0.0),
            precipitation=item.precipitation,
            polygon = [],
            temperature=item.temperature,
            dewpoint=item.dewpoint,
        )
        for i, item in enumerate(data)
    ]

    preds = run_batch_inference(rows)

    return {
        "prediction_window": "72 Hours",
        "count": len(preds),
        "results": [
            {
                "fire_probability": p.fire_probability,
                "risk_level": p.risk_level,
            }
            for p in preds
        ],
    }

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

boundary_path = Path(__file__).parent / "data" / "bc_land.geojson"
with open(boundary_path, "r", encoding="utf-8") as f:
    geo = json.load(f)

# Handles FeatureCollection or single Feature
if geo.get("type") == "FeatureCollection":
    bc_geom = shape(geo["features"][0]["geometry"])
elif geo.get("type") == "Feature":
    bc_geom = shape(geo["geometry"])
else:
    bc_geom = shape(geo)

ml_assets["bc_land_geom"] = bc_geom
ml_assets["bc_land_prepared"] = prep(bc_geom)