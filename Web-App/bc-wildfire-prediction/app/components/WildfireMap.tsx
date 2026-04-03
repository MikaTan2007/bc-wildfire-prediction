"use client";

import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, Polygon, Popup } from "react-leaflet";

type GridPoint = {
  lat: number;
  lon: number;
};

type GridPrediction = {
  grid_id: string;
  centroid: GridPoint;
  polygon: GridPoint[];
  precipitation: number;
  temperature: number;
  dewpoint: number;
  fire_probability: number;
  risk_level: "Low" | "Moderate" | "High";
};

type GridPredictionResponse = {
  generated_at: string;
  cell_count: number;
  prediction_window: string;
  weather_source: string;
  cells: GridPrediction[];
};

function getCellColor(probability: number): string {
  if (probability >= 80) return "#b10026";
  if (probability >= 60) return "#fc4e2a";
  if (probability >= 40) return "#fd8d3c";
  if (probability >= 20) return "#feb24c";
  return "#31a354";
}

export default function WildfireMap() {
  const [data, setData] = useState<GridPredictionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchedAt = useMemo(() => {
    if (!data?.generated_at) return "";
    return new Date(data.generated_at).toLocaleString();
  }, [data?.generated_at]);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    async function load() {
      try {
        setLoading(true);
        setError(null);

        const resp = await fetch("/api/grid-predict", {
          cache: "no-store",
          signal: controller.signal,
        });

        const payload = await resp.json();

        if (!resp.ok) {
          throw new Error(payload?.detail || payload?.error || "Request failed");
        }

        if (active) setData(payload as GridPredictionResponse);
      } catch (e) {
        if (!active) return;
        if (e instanceof DOMException && e.name === "AbortError") return;
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        if (active) setLoading(false);
      }
    }

    load();

    return () => {
      active = false;
      controller.abort();
    };
  }, []);

  if (loading) {
    return <div className="status">Loading wildfire grid predictions...</div>;
  }

  if (error) {
    return <div className="status error">Error: {error}</div>;
  }

  if (!data) {
    return <div className="status">No data available.</div>;
  }

  return (
    <div className="map-page">
      <div className="header">
        <h1>BC Wildfire Likelihood Within 72 Hours</h1>
        <p>
          Cells: {data.cell_count} | Source: {data.weather_source} | Updated: {fetchedAt}
        </p>
      </div>

      <div className="legend">
        <div><span style={{ background: "#31a354" }} />0-19%</div>
        <div><span style={{ background: "#feb24c" }} />20-39%</div>
        <div><span style={{ background: "#fd8d3c" }} />40-59%</div>
        <div><span style={{ background: "#fc4e2a" }} />60-79%</div>
        <div><span style={{ background: "#b10026" }} />80-100%</div>
      </div>

      <MapContainer center={[54.5, -125.0]} zoom={5} minZoom={4} className="map-container" scrollWheelZoom>
        <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {data.cells.map((cell) => {
          const positions = cell.polygon.map((pt) => [pt.lat, pt.lon] as [number, number]);
          const color = getCellColor(cell.fire_probability);

          return (
            <Polygon
              key={cell.grid_id}
              positions={positions}
              pathOptions={{ color, weight: 0.4, fillColor: color, fillOpacity: 0.55 }}
            >
              <Popup>
                <strong>{cell.grid_id}</strong>
                <br />
                Probability: {cell.fire_probability}%
                <br />
                Temp: {cell.temperature.toFixed(2)} C
                <br />
                Dewpoint: {cell.dewpoint.toFixed(2)} C
                <br />
                Precip: {cell.precipitation.toFixed(2)} mm
              </Popup>
            </Polygon>
          );
        })}
      </MapContainer>
    </div>
  );
}