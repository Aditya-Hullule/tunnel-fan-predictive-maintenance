import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

from scipy.stats import kurtosis


st.set_page_config(
    page_title="Tunnel Fan Live Monitoring",
    page_icon="⚙️",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load("models/model_v2.pkl")


model = load_model()


def extract_time_features(signal):

    rms = np.sqrt(np.mean(signal ** 2))
    std = np.std(signal)
    kurt = kurtosis(signal, fisher=False)
    peak_to_peak = np.ptp(signal)
    crest_factor = np.max(np.abs(signal)) / (rms + 1e-10)

    return {
        "rms": rms,
        "std": std,
        "kurtosis": kurt,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor
    }


def extract_frequency_features(signal):

    signal = signal - np.mean(signal)

    n = len(signal)

    fft_values = np.fft.rfft(signal)

    magnitude = np.abs(fft_values) / n

    frequencies = np.fft.rfftfreq(n, d=1.0)

    if len(magnitude) > 1:
        magnitude[0] = 0

    total_magnitude = np.sum(magnitude) + 1e-10

    dominant_index = np.argmax(magnitude)

    dominant_frequency = frequencies[dominant_index]

    dominant_amplitude = magnitude[dominant_index]

    spectral_centroid = (
        np.sum(frequencies * magnitude)
        / total_magnitude
    )

    spectral_bandwidth = np.sqrt(
        np.sum(
            ((frequencies - spectral_centroid) ** 2)
            * magnitude
        )
        / total_magnitude
    )

    spectral_energy = np.sum(magnitude ** 2)

    return {
        "dominant_frequency": dominant_frequency,
        "dominant_amplitude": dominant_amplitude,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_energy": spectral_energy
    }


def extract_features(data):

    features = {}

    for channel in [1, 2, 3, 4, 5, 6]:

        signal = data.iloc[:, channel].to_numpy(
            dtype=np.float64
        )

        time_features = extract_time_features(signal)

        for name, value in time_features.items():
            features[f"ch{channel}_{name}"] = value

        frequency_features = extract_frequency_features(signal)

        for name, value in frequency_features.items():
            features[f"ch{channel}_{name}"] = value

    return pd.DataFrame([features])


def classify_status(prediction, confidence):

    if prediction == "normal":
        return "NORMAL", "normal"

    if confidence >= 80:
        return "FAULT DETECTED", "high"

    return "ATTENTION REQUIRED", "medium"


st.title("⚙️ Tunnel Fan Predictive Maintenance")

st.caption(
    "Real-Time Predictive Maintenance Simulation"
)

st.info(
    "Simulation mode: recorded vibration data is processed "
    "as a simulated incoming sensor stream."
)

st.divider()


data_root = "data/raw/mafaulda"


@st.cache_data
def get_files():

    import os

    files = []

    for root, directories, filenames in os.walk(data_root):

        for filename in filenames:

            if filename.lower().endswith(".csv"):

                full_path = os.path.join(
                    root,
                    filename
                )

                files.append(full_path)

    return sorted(files)


files = get_files()


if not files:

    st.error(
        "No vibration CSV files were found."
    )

    st.stop()


selected_file = st.selectbox(
    "Select recorded vibration data",
    files,
    format_func=lambda x: x.replace(
        data_root + "\\",
        ""
    )
)


window_size = st.selectbox(
    "Simulation window size",
    [50000, 75000, 100000],
    index=0
)


speed = st.slider(
    "Simulation speed",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1
)


start_button = st.button(
    "▶ Start Live Simulation",
    type="primary"
)


if start_button:

    data = pd.read_csv(
        selected_file,
        header=None
    )

    if data.shape[1] < 7:

        st.error(
            "Invalid vibration file."
        )

        st.stop()


    total_samples = len(data)

    windows = []

    for start in range(
        0,
        total_samples,
        window_size
    ):

        end = min(
            start + window_size,
            total_samples
        )

        windows.append(
            data.iloc[start:end]
        )


    st.success(
        f"Loaded {total_samples:,} vibration samples "
        f"and created {len(windows)} monitoring windows."
    )


    status_placeholder = st.empty()

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    prediction_placeholder = metrics_col1.empty()
    confidence_placeholder = metrics_col2.empty()
    window_placeholder = metrics_col3.empty()

    st.subheader("Live Vibration Signal")

    chart_placeholder = st.empty()

    st.subheader("Prediction History")

    history_placeholder = st.empty()

    history = []

    for window_number, window in enumerate(
        windows,
        start=1
    ):

        start_time = time.perf_counter()

        feature_data = extract_features(
            window
        )

        feature_time = (
            time.perf_counter()
            - start_time
        )


        model_start = time.perf_counter()

        prediction = model.predict(
            feature_data
        )[0]

        probabilities = model.predict_proba(
            feature_data
        )[0]

        model_time = (
            time.perf_counter()
            - model_start
        )


        total_time = (
            time.perf_counter()
            - start_time
        )


        confidence = (
            np.max(probabilities)
            * 100
        )


        status, level = classify_status(
            prediction,
            confidence
        )


        if prediction == "normal":

            status_placeholder.success(
                f"🟢 FAN STATUS: {status}"
            )

        elif confidence >= 80:

            status_placeholder.error(
                f"🔴 FAN STATUS: {status}"
            )

        else:

            status_placeholder.warning(
                f"🟡 FAN STATUS: {status}"
            )


        prediction_placeholder.metric(
            "Current Condition",
            prediction.replace(
                "-",
                " "
            ).title()
        )


        confidence_placeholder.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        window_placeholder.metric(
            "Monitoring Window",
            f"{window_number}/{len(windows)}"
        )


        preview_size = min(
            3000,
            len(window)
        )


        signal = window.iloc[
            :preview_size,
            1
        ].reset_index(drop=True)


        chart_data = pd.DataFrame({
            "Vibration": signal
        })


        chart_placeholder.line_chart(
            chart_data
        )


        history.append({
            "Window": window_number,
            "Prediction": prediction.replace(
                "-",
                " "
            ).title(),
            "Confidence": f"{confidence:.2f}%",
            "Processing": f"{total_time * 1000:.1f} ms"
        })


        history_placeholder.dataframe(
            pd.DataFrame(history),
            use_container_width=stretch,
            hide_index=True
        )


        time.sleep(
            max(
                0.1,
                1.0 / speed
            )
        )


    st.success(
        "✅ Simulation completed."
    )

else:

    st.markdown(
        """
        ### System Architecture

        **Recorded vibration → simulated sensor stream → 
        60-feature extraction → Random Forest → 
        continuous fault prediction**

        ### Future Hardware Integration

        The simulated data source can later be replaced by
        a physical vibration sensor connected to an edge device.
        """
    )