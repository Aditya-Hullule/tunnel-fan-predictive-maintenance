# AI-Based Predictive Maintenance System for Tunnel Ventilation Fans

An AI-based prototype for vibration-based fault diagnosis of tunnel ventilation fans using Machine Learning. The system extracts time-domain and frequency-domain features from vibration data and uses a Random Forest classifier to identify fan health conditions.

It includes two Streamlit demos:
- **Demo 1 (`app.py`)**: Upload a vibration CSV file to perform single-file fault prediction and analysis.
- **Demo 2 (`live_demo.py`)**: Simulates a continuous live sensor stream for real-time monitoring.

---

## Prerequisites

- Python 3.9 or newer
- Git

---

## Quick Start

### 1. Clone & Navigate to Project

```bash
git clone https://github.com/Aditya-Hullule/tunnel-fan-predictive-maintenance.git
cd tunnel-fan-predictive-maintenance
```

### 2. Create & Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Demos

### Demo 1 — Vibration Fault Prediction (Upload & Predict)

```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser and upload a vibration CSV file.

### Demo 2 — Simulated Live Monitoring

```bash
streamlit run live_demo.py
```
Open `http://localhost:8501` in your browser, select a data file and window size, then start the stream simulation.

*(Press `Ctrl + C` in the terminal to stop either application.)*
