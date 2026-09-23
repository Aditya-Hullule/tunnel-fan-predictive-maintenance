import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

from scipy.stats import kurtosis


st.set_page_config(
    page_title="Tunnel Fan Predictive Maintenance",
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

    spectral_centroid = np.sum(
        frequencies * magnitude
    ) / total_magnitude

    spectral_bandwidth = np.sqrt(
        np.sum(
            ((frequencies - spectral_centroid) ** 2) * magnitude
        ) / total_magnitude
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

    vibration_columns = [1, 2, 3, 4, 5, 6]

    features = {}

    for column in vibration_columns:

        signal = data.iloc[:, column].to_numpy(
            dtype=np.float64
        )

        time_features = extract_time_features(signal)

        for name, value in time_features.items():
            features[f"ch{column}_{name}"] = value

        frequency_features = extract_frequency_features(signal)

        for name, value in frequency_features.items():
            features[f"ch{column}_{name}"] = value

    return pd.DataFrame([features])


def get_status(prediction):

    if prediction == "normal":
        return "NORMAL", "System operating normally"

    return "FAULT DETECTED", f"{prediction.replace('-', ' ').title()} detected"


st.title("⚙️ Tunnel Fan Predictive Maintenance System")

st.markdown(
    """
    ### AI-Based Vibration Fault Detection

    Upload a tunnel fan vibration CSV file to analyze its condition
    using the trained Random Forest predictive-maintenance model.
    """
)

st.divider()


uploaded_file = st.file_uploader(
    "Upload vibration CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        data = pd.read_csv(
            uploaded_file,
            header=None
        )

        st.success(
            f"File loaded successfully — {len(data):,} samples"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Samples",
                f"{len(data):,}"
            )

        with col2:
            st.metric(
                "Channels",
                "6"
            )

        with col3:
            st.metric(
                "Model Features",
                "60"
            )

        st.divider()

        if data.shape[1] < 7:

            st.error(
                "Invalid CSV: expected at least 7 columns."
            )

        else:

            with st.spinner("Analyzing vibration signal..."):

                start_time = time.perf_counter()

                feature_data = extract_features(data)

                feature_time = (
                    time.perf_counter() - start_time
                )

                start_model = time.perf_counter()

                prediction = model.predict(
                    feature_data
                )[0]

                probabilities = model.predict_proba(
                    feature_data
                )[0]

                model_time = (
                    time.perf_counter() - start_model
                )

                total_time = (
                    time.perf_counter() - start_time
                )

            status, message = get_status(
                prediction
            )

            st.subheader("Prediction Result")

            if prediction == "normal":

                st.success(
                    f"✅ {status}"
                )

            else:

                st.error(
                    f"⚠️ {status}"
                )

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.markdown(
                    f"### Detected Condition"
                )

                st.markdown(
                    f"## {prediction.replace('-', ' ').title()}"
                )

            with result_col2:

                confidence = np.max(probabilities) * 100

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

            st.info(message)

            st.divider()

            st.subheader("Fault Probabilities")

            probability_df = pd.DataFrame({
                "Condition": model.classes_,
                "Probability": probabilities * 100
            })

            probability_df = probability_df.sort_values(
                "Probability",
                ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Condition")
            )

            st.dataframe(
                probability_df.style.format(
                    {
                        "Probability": "{:.2f}%"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader("Processing Performance")

            time_col1, time_col2, time_col3 = st.columns(3)

            with time_col1:
                st.metric(
                    "Feature Extraction",
                    f"{feature_time * 1000:.2f} ms"
                )

            with time_col2:
                st.metric(
                    "ML Inference",
                    f"{model_time * 1000:.2f} ms"
                )

            with time_col3:
                st.metric(
                    "Total Processing",
                    f"{total_time * 1000:.2f} ms"
                )

            st.divider()

            st.subheader("Extracted Features")

            display_features = feature_data.T.reset_index()

            display_features.columns = [
                "Feature",
                "Value"
            ]

            st.dataframe(
                display_features,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader("Vibration Signal Preview")

            channel = st.selectbox(
                "Select vibration channel",
                [1, 2, 3, 4, 5, 6]
            )

            signal = data.iloc[:, channel]

            preview_size = min(
                5000,
                len(signal)
            )

            signal_preview = pd.DataFrame({
                f"Channel {channel}": signal.iloc[:preview_size].values
            })

            st.line_chart(
                signal_preview
            )

    except Exception as e:

        st.error(
            f"Error processing file: {e}"
        )


else:

    st.info(
        "Upload a vibration CSV file to start the analysis."
    )

    st.markdown(
        """
        ### Model Information

        **Algorithm:** Random Forest

        **Features:** 60

        **Channels:** 6

        **Time-domain features:** 30

        **Frequency-domain features:** 30

        **Training samples:** 1,951
        """
    )