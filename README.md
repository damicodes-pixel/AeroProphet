# AeroProphet

### Predicting and Visualizing Flight Disruption Propagation

AeroProphet is an aviation machine-learning system designed to model, predict, and visualize flight disruptions across interconnected airline operations.

Rather than treating every flight as an isolated prediction, AeroProphet incorporates flight context, aircraft history, previous-leg delays, turnaround conditions, cancellations, diversions, and temporal information to investigate how disruptions propagate through aviation operations.

The system combines a tabular XGBoost model for broad flight-state prediction with a GRU sequence model specifically designed to capture aircraft-level temporal propagation.

The project also includes a visualization and simulation software layer that uses model predictions to replay aircraft operations and make predicted disruption propagation observable over time.

> **AeroProphet asks a more interesting question than “Will this flight be late?”**
>
> **“Given what has already happened to this aircraft and its surrounding operational context, what happens next?”**

---

## System Architecture

```text
                    BTS / TranStats Data
                             |
                             v
                  +----------------------+
                  |   Feature Engineering |
                  +----------+-----------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
     +-----------------+           +-----------------+
     |  XGBoost Model  |           |   GRU Model     |
     |                 |           |                 |
     | Flight-state    |           | Aircraft        |
     | prediction      |           | history         |
     |                 |           |                 |
     | 97+ features    |           | 4 previous legs |
     +--------+--------+           | x 5 features    |
              |                    +--------+--------+
              |                             |
              +--------------+--------------+
                             |
                             v
                  +-----------------------+
                  |  Prediction Pipeline  |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Simulation / Replay   |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Visualization Software|
                  |                       |
                  | Aircraft movement      |
                  | Disruption states      |
                  | Model predictions      |
                  | Temporal propagation   |
                  +-----------------------+
```

---

## Problem

Flight delays are not independent events.

A flight can arrive late because of weather, congestion, mechanical problems, crew constraints, or other operational disruptions. That late aircraft may then operate another flight, reducing its available turnaround time and increasing the probability that the next departure will also be delayed.

This creates a chain:

```text
Flight A
   |
   | arrives late
   v
Aircraft turnaround
   |
   | reduced recovery time
   v
Flight B
   |
   | departs late
   v
Flight C
   |
   v
Network disruption
```

Traditional tabular prediction can identify relationships between individual flight features and outcomes.

AeroProphet additionally represents the recent operational history of the aircraft itself.

---

## Dataset

The modeling pipeline uses publicly available Bureau of Transportation Statistics flight data.

The final modeling dataset contains:

- approximately 1.5M flight records in the active modeling window
- flight-level operational information
- aircraft tail numbers
- scheduled departure and arrival information
- departure and arrival delays
- cancellation status
- diversion status
- route information
- distance and distance groups
- temporal information

### Core flight features

```text
Month
DayofMonth
DayOfWeek
FlightDate
Reporting_Airline
Tail_Number
Flight_Number_Reporting_Airline
Origin
OriginState
Dest
DestState
CRSDepTime
DepDelayMinutes
DepDel15
DepTimeBlk
CRSArrTime
ArrDelayMinutes
ArrDel15
ArrTimeBlk
Cancelled
Diverted
CRSElapsedTime
Distance
DistanceGroup
```

---

## Operational Context

AeroProphet constructs aircraft-history features to represent what happened to an aircraft before its current flight.

Across the processed aircraft history, the system generates features including:

- `plane_prev_arr_delay`
- `plane_prev_known`
- `plane_prev_cancelled`
- `plane_prev_diverted`
- `plane_turnaround`
- `plane_leg_of_day`

It also computes rolling airport, route, and state-route disruption context using recent historical flights.

These features are designed to encode the idea that delay propagation is not purely a property of a single flight, but a property of the recent operational state surrounding the aircraft and network.

### Operational context statistics

Observed aircraft-history statistics from the processed dataset:

```text
Previous arrival delay
Mean:       12.44 min
Median:      0.00 min
75th pct:    7.00 min
Maximum:  4336.00 min

Previous-leg information available
Mean: 80%

Previous-leg cancellation rate
Mean: 1%

Previous-leg diversion rate
Mean: <1%

Turnaround time
Mean:   194.66 min
Median:  72.00 min
75th pct: 305.00 min
Maximum: 720.00 min
```

---

## Key Evaluation Statistics

### Dataset and split

```text
Total flight records:       1,500,000
Model columns:              106
Training rows:              1,198,721
Testing rows:               301,279
Temporal split date:        2025-10-19
Evaluation method:          chronological 80/20 split by PredictionTime
```

### XGBoost flight-state model

```text
Accuracy:          0.8405
Balanced Accuracy: 0.6049
```

Classification summary:

```text
Class           Precision  Recall  F1
On Time         0.86       0.96   0.90
Delayed         0.76       0.48   0.59
Cancelled       0.76       0.37   0.50
```

Confusion matrix:

```text
                 Predicted
              On Time  Delayed  Cancelled
Actual On Time   218211    8921       237
Actual Delayed    35055   32963       403
Actual Cancelled   1947    1494      2048
```

### GRU propagation model

```text
Training examples:  1,372,069
Test examples:      343,071
Sequence length:    4 flights
Sequence features:  5 per leg
GRU hidden size:    32
Optimizer:          Adam
Learning rate:      0.001
Epochs:            5
Loss:              BCEWithLogitsLoss
```

Performance:

```text
Epoch 1: Train Loss 0.4233 | Test ROC-AUC 0.804
Epoch 2: Train Loss 0.3802 | Test ROC-AUC 0.810
Epoch 3: Train Loss 0.3765 | Test ROC-AUC 0.814
Epoch 4: Train Loss 0.3740 | Test ROC-AUC 0.818
Epoch 5: Train Loss 0.3719 | Test ROC-AUC 0.819
```

Final GRU evaluation:

```text
GRU ROC-AUC:              0.819
GRU PR-AUC:               0.725
Previous-arrival baseline: 0.578 ROC-AUC
```

### Context ablation

```text
Without context features: 0.6169 ROC-AUC
With context features:    0.8241 ROC-AUC
Difference:                +0.207 ROC-AUC
```

### Additional operational models

#### Departure delay model (`DepDel15`)

```text
Accuracy: 0.8447
ROC-AUC:  0.8241
PR-AUC:   0.6989
```

#### Cancellation model

```text
Accuracy:   0.9862
ROC-AUC:    0.9239
PR-AUC:     0.5535
Base rate:  1.855%
```

#### Diversion model

```text
Accuracy:   0.99784
ROC-AUC:    0.7232
PR-AUC:     0.0161
Base rate:  0.2155%
```

#### Delay regression (`DepDelayMinutes`)

```text
Model MAE:            20.97 minutes
Median baseline MAE:  18.06 minutes
Mean actual delay:    18.06 minutes
Median actual delay:   0.00 minutes
Maximum actual delay: 2601.00 minutes
```

The regression result is included as a fair evaluation outcome: the model did not beat the median baseline on MAE, and this honest result is retained rather than hidden.

---

## XGBoost Flight-State Model

The first major modeling component uses XGBoost to predict flight operational state.

The target contains three classes:

```text
0 -> On Time
1 -> Delayed
2 -> Cancelled
```

### Dataset split

Train/test split follows a chronological approach to avoid leakage:

```text
Total:       model window size varies by run
Training:    first 80%
Testing:     final 20%
```

The split is time-based and ordered by `PredictionTime`, preventing later flights from being used to train predictions for earlier periods.

### Modeling approach

The main XGBoost model uses:

- tree-based gradient boosting
- categorical handling for operational fields
- a rich engineered feature space combining static flight features and temporal context
- time-based validation rather than random shuffling

This creates a broad flight-state prediction layer that captures disruption risk at the flight and network-context level.

---

## GRU Propagation Model

The second major component specifically models aircraft-level temporal history.

Instead of looking only at the current flight, the model receives the previous four legs operated by the same aircraft.

```text
Previous Leg 4
      |
Previous Leg 3
      |
Previous Leg 2
      |
Previous Leg 1
      |
Current Flight
```

The sequence contains five features for each previous leg:

1. Previous arrival delay
2. Previous departure delay
3. Whether the previous flight had landed
4. Whether the previous flight had departed
5. Turnaround gap

The resulting sequence contains:

```text
4 previous legs x 5 features
```

The sequence is ordered chronologically from oldest to newest.

---

## PropagationNet

The GRU architecture is intentionally compact:

```text
Input sequence
4 x 5
   |
   v
GRU
32 hidden units
   |
   v
Final hidden state
   |
   +----------------+
   |                |
   | Current flight |
   | 4 flat features |
   |                |
   +-------+--------+
           |
           v
      Concatenation
           |
           v
       Linear 32
           |
          ReLU
           |
           v
       Linear 1
           |
           v
   Delay probability
```

### Current-flight features

The GRU representation is combined with four normalized features describing the current flight:

- scheduled departure time
- day of week
- distance
- scheduled elapsed time

The model therefore combines:

> What has happened to this aircraft?

with:

> What is this flight?

---

## GRU Training

The deep-learning model is trained as a temporal sequence classification task.

```text
Target:                 DepDel15
Positive class:         15+ minute departure delay
Sequence length:        4 flights
Sequence features:      5 per leg
GRU hidden size:        32
Optimizer:              Adam
Learning rate:          0.001
Epochs:                 5
Train batch size:       2048
Test batch size:        8192
Loss:                   BCEWithLogitsLoss
```

The sequence model is designed to capture whether recent operational history of the same aircraft is predictive of a future disruption.

---

## GRU Results

The temporal propagation model is evaluated in terms of ROC-AUC and PR-AUC relative to a simple previous-delay baseline.

This provides evidence that recent aircraft behavior carries predictive information beyond a single prior delay value.

---

## Context Ablation

AeroProphet tests the contribution of operational context features.

The purpose is to measure how much predictive information is added when the model receives context instead of relying primarily on isolated flight attributes.

This comparison is important because it distinguishes:

- a single-flight delay classifier
- from a true context-aware operational forecasting model

---

## Additional Flight-State Models

In addition to the main multi-class model, the codebase trains targeted models for operational events:

### Departure Delay

- `DepDel15` classifier
- time-based split
- ROC-AUC / PR-AUC evaluation
- classification report with precision, recall, and F1

### Cancellation

- `Cancelled` classifier
- highly imbalanced target
- evaluated with ROC-AUC, PR-AUC, and base rate

### Diversion

- `Diverted` classifier
- highly imbalanced rare-event target
- includes honest reporting of imbalance and threshold effects

### Delay Regression

- `DepDelayMinutes` regressor
- `ArrDelayMinutes` regressor
- MAE evaluation with a median baseline

This gives a more complete operational picture than a single binary delay label alone.

---

## Visualization and Simulation Software

AeroProphet includes a dedicated visualization and simulation software layer for turning model predictions into an observable aviation environment.

Rather than stopping at:

```text
Model -> CSV -> README
```

The system exports predictions that can be replayed through the visualization software to represent aircraft operations and predicted disruption states over time.

The visualization layer is designed around the same temporal structure used by the models:

```text
Flight history
     |
     v
Aircraft state
     |
     v
Model prediction
     |
     v
Simulation timeline
     |
     v
Visual aircraft/network state
```

The GRU prediction pipeline exports:

```text
data/sim/gru_predictions.csv
```

and the XGBoost pipeline exports:

```text
data/sim/xgb_predictions.csv
```

These are merged and exported into a replay JSON used by the visualization interface.

---

## Project Pipeline

```text
BTS Flight Data
      |
      v
Data Processing
      |
      v
Feature Engineering
      |
      +----------------------+
      |                      |
      v                      v
XGBoost                 Aircraft History
Flight-State            Sequence Construction
Prediction                   |
      |                      v
      |                    GRU
      |                      |
      +----------+-----------+
                 |
                 v
          Model Predictions
                 |
                 v
       Simulation Data Export
                 |
                 v
      Visualization Software
                 |
                 v
       Temporal Replay of
      Aviation Disruptions
```

---

## Model Outputs

Saved trained artifacts include:

```text
models/
├── air_xgb_model.json
├── depdel15_model.json
├── cancelled_model.json
├── diverted_model.json
├── depdelay_minutes_model.json
├── arrdelay_minutes_model.json
├── propagation_net.pt
```

Simulation predictions are exported to:

```text
data/sim/
├── xgb_predictions.csv
├── gru_predictions.csv
```

---

## Project Structure

```text
AeroProphet/
|
├── README.md
├── requirements.txt
├── pyproject.toml
|
├── notebooks/
|   ├── base_model.ipynb
|   └── deep_learning.ipynb
|
├── scripts/
|   ├── scrape.py
|   ├── build_replay.py
|   └── serve.py
|
├── src/
|   └── aeroprophet/
|       └── __init__.py
|
├── data/
|   ├── raw/
|   |   └── bts_raw_data/
|   ├── processed/
|   └── sim/
|       ├── xgb_predictions.csv
|       └── gru_predictions.csv
|
├── models/
|   └── *.json / *.pt
|
├── web/
|   ├── index.html
|   └── replay.json
|
├── docs/
|
├── tests/
|
└── .venv/
```

---

## Technology

### Machine Learning

- XGBoost
- PyTorch
- GRU / recurrent neural networks

### Data Processing

- Polars
- Pandas
- NumPy

### Evaluation

- ROC-AUC
- PR-AUC
- MAE
- Precision
- Recall
- F1
- Confusion matrices

### Data

- Bureau of Transportation Statistics
- BTS / TranStats flight records

### Visualization

- custom aviation simulation software
- temporal prediction replay
- aircraft-level state visualization

---

## Future Development

Potential extensions include:

### Network-Level Propagation

Move beyond individual aircraft histories toward an explicit graph representing:

```text
Aircraft
Airport
Route
Flight
Time
```

### Airport State Modeling

Represent airport-level congestion and accumulated disruption as dynamic state.

### Multi-Step Forecasting

Predict disruption several legs into the future rather than only the next flight.

### Interactive Simulation

Allow users to manipulate operational conditions and observe how disruptions propagate through the simulated network.

### Reinforcement Learning

Use the simulation environment as a foundation for training an agent to interact with and respond to a dynamic aviation network.

---

## Summary

AeroProphet combines:

```text
Tabular ML
    +
Temporal Sequence Modeling
    +
Aircraft Operational History
    +
Context Features
    +
Simulation
    +
Visualization
```

The system moves beyond isolated flight-delay classification toward modeling aviation as a dynamic operational system, with machine-learning predictions feeding directly into a temporal simulation and visualization environment.

---

## Running AeroProphet on Your Own Machine

### 1) Clone the project

```bash
git clone <your-repo-url>
cd AeroProphet
```

### 2) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, if you want the package installed in editable mode:

```bash
pip install -e .
```

### 4) Run the notebooks

Open the notebooks in Jupyter Lab:

```bash
jupyter lab
```

Then run:

- `notebooks/base_model.ipynb` for the XGBoost and flight-state modeling pipeline
- `notebooks/deep_learning.ipynb` for the GRU propagation model

### 5) Build the simulation replay

After model predictions have been exported, generate the replay data:

```bash
python scripts/build_replay.py --days 2 --max-flights 12000
```

This merges the XGBoost and GRU prediction files and writes the replay payload to `web/replay.json`.

### 6) Start the local simulator UI

```bash
python scripts/serve.py
```

This serves the dashboard locally at:

```text
http://localhost:8000
```

### 7) Optional: build a fresh data fetch

If you want to refresh the BTS raw dataset:

```bash
python scripts/scrape.py
```

The project expects the raw data folder to be available in the expected BTS structure before model training begins.

---

## Author

**Dami Ogunsuyi**

Computer Science @ UMBC
Machine Learning • AI Systems • Aviation Intelligence
