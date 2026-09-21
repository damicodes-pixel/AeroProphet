"""Merge both models' held-out predictions into web/replay.json for the sim."""
import argparse, json
from pathlib import Path
import pandas as pd

# (lat, lon) for the big continental-US airports. Flights to other airports are skipped.
AIRPORTS = {
    "ATL": (33.64, -84.43), "LAX": (33.94, -118.41), "ORD": (41.98, -87.90), "DFW": (32.90, -97.04),
    "DEN": (39.86, -104.67), "JFK": (40.64, -73.78), "SFO": (37.62, -122.38), "SEA": (47.45, -122.31),
    "LAS": (36.08, -115.15), "MCO": (28.43, -81.31), "EWR": (40.69, -74.17), "CLT": (35.21, -80.94),
    "PHX": (33.43, -112.01), "IAH": (29.98, -95.34), "MIA": (25.79, -80.29), "BOS": (42.36, -71.01),
    "MSP": (44.88, -93.22), "FLL": (26.07, -80.15), "DTW": (42.21, -83.35), "PHL": (39.87, -75.24),
    "LGA": (40.78, -73.87), "BWI": (39.18, -76.67), "SLC": (40.79, -111.98), "SAN": (32.73, -117.19),
    "IAD": (38.95, -77.46), "DCA": (38.85, -77.04), "MDW": (41.79, -87.75), "TPA": (27.98, -82.53),
    "PDX": (45.59, -122.60), "STL": (38.75, -90.37), "BNA": (36.12, -86.68), "AUS": (30.19, -97.67),
    "DAL": (32.85, -96.85), "HOU": (29.65, -95.28), "OAK": (37.72, -122.22), "MSY": (29.99, -90.26),
    "RDU": (35.88, -78.79), "SMF": (38.70, -121.59), "SJC": (37.36, -121.93), "SNA": (33.68, -117.87),
    "MCI": (39.30, -94.71), "CLE": (41.41, -81.85), "IND": (39.72, -86.29), "PIT": (40.49, -80.23),
    "CMH": (40.00, -82.89), "SAT": (29.53, -98.47), "CVG": (39.05, -84.67), "MKE": (42.95, -87.90),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=2.0, help="how many of the latest days to replay")
    ap.add_argument("--max-flights", type=int, default=12000)
    args = ap.parse_args()

    key = ["Tail_Number", "PredictionTime"]
    xgb = pd.read_csv("sim_data/xgb_predictions.csv", parse_dates=["PredictionTime"]).drop_duplicates(key)
    gru = pd.read_csv("sim_data/gru_predictions.csv", parse_dates=["PredictionTime"]).drop_duplicates(key)

    df = xgb.merge(gru, on=key, how="inner")
    df = df[df["Origin"].isin(AIRPORTS) & df["Dest"].isin(AIRPORTS)]
    if df.empty:
        raise SystemExit("No flights in common between the two prediction files. Re-run both export cells.")

    df = df[df["PredictionTime"] > df["PredictionTime"].max() - pd.Timedelta(days=args.days)]
    if len(df) > args.max_flights:
        df = df.sample(args.max_flights, random_state=1)
    df = df.sort_values("PredictionTime")
    start = df["PredictionTime"].min().floor("h")

    flights = [
        {
            "o": r.Origin, "d": r.Dest, "n": r.Tail_Number,
            "t": int((r.PredictionTime - start).total_seconds() // 60),
            "e": max(20, int(r.CRSElapsedTime)) if pd.notna(r.CRSElapsedTime) else 120,
            "a": int(r.DepDelayMinutes) if pd.notna(r.DepDelayMinutes) else 0,
            "x": round(float(r.xgb_prob), 3), "g": round(float(r.gru_prob), 3),
        }
        for r in df.itertuples()
    ]
    used = {c for f in flights for c in (f["o"], f["d"])}
    out = {
        "start": start.isoformat(),
        "airports": {c: {"lat": AIRPORTS[c][0], "lon": AIRPORTS[c][1]} for c in used},
        "flights": flights,
    }
    Path("web").mkdir(exist_ok=True)
    Path("web/replay.json").write_text(json.dumps(out))
    print(f"Wrote web/replay.json: {len(flights)} flights, {len(used)} airports, starting {start}")


if __name__ == "__main__":
    main()