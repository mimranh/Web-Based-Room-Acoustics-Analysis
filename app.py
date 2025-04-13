from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scipy import signal  # Imported for potential signal processing tasks

app = Flask(__name__)

# Define the folder for uploaded files and allowed file extensions.
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension (only '.wav' is allowed).
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_rt60(audio_data, sample_rate):
    """
    Calculate RT60 (Reverberation Time) using a simplified version of Schroeder's method.

    Steps:
    1. If the audio data is multi-channel, average to convert it to mono.
    2. Square the audio signal to compute the energy.
    3. Compute the Energy Decay Curve (EDC) by taking the reverse cumulative sum.
    4. Normalize the EDC and convert it to decibels.
    5. Identify the time indices where the energy has decayed by 5 dB and 35 dB.
    6. Extrapolate these values to estimate RT60.
    """
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    squared_signal = audio_data**2
    # Compute the Energy Decay Curve by reverse cumulative summing the squared signal.
    edc = np.flip(np.cumsum(np.flip(squared_signal)))
    edc = edc / np.max(edc)  # Normalize the EDC to a maximum of 1
    edc_db = 10 * np.log10(
        edc + 1e-10
    )  # Convert to decibels; small constant avoids log(0)

    # Find the index where the decay reaches -5 dB and -35 dB
    idx_5 = np.where(edc_db <= -5)[0]
    idx_35 = np.where(edc_db <= -35)[0]

    if len(idx_5) == 0 or len(idx_35) == 0:
        # If we cannot find the required decay points, return None
        return None
    else:
        # Use the first occurrence of each decay threshold
        idx_5 = idx_5[0]
        idx_35 = idx_35[0]
        time_5 = idx_5 / sample_rate  # Convert sample index to time in seconds
        time_35 = idx_35 / sample_rate
        # Extrapolate to get RT60: multiply the time difference by 2 (i.e., 60 dB decay extrapolation)
        rt60 = 2 * (time_35 - time_5)
        return rt60


def calculate_clarity(audio_data, sample_rate, ms_threshold=80):
    """
    Calculate clarity index (often denoted as C80) for the audio signal.

    Steps:
    1. Determine the number of samples corresponding to the provided threshold (default is 80 ms).
    2. Compute the 'early' energy (energy within the threshold) and 'late' energy (remaining energy).
    3. Calculate clarity as 10 * log10(early_energy / late_energy).
    """
    threshold_samples = int(ms_threshold * sample_rate / 1000)
    early_energy = np.sum(audio_data[:threshold_samples] ** 2)
    late_energy = np.sum(audio_data[threshold_samples:] ** 2)
    clarity = 10 * np.log10(early_energy / (late_energy + 1e-10))
    return clarity


def calculate_c60(audio_data, sample_rate, ms_threshold=60):
    """
    Calculate the clarity index for speech (C60), which uses a 60 ms threshold.

    The process is similar to calculate_clarity, but with a 60 ms integration time.
    """
    threshold_samples = int(ms_threshold * sample_rate / 1000)
    early_energy = np.sum(audio_data[:threshold_samples] ** 2)
    late_energy = np.sum(audio_data[threshold_samples:] ** 2)
    c60 = 10 * np.log10(early_energy / (late_energy + 1e-10))
    return c60


def calculate_edt(audio_data, sample_rate):
    """
    Calculate the Early Decay Time (EDT).

    EDT is estimated by measuring the time it takes for the energy to drop by 10 dB
    (from 0 dB to -10 dB) on the Energy Decay Curve (EDC) and then extrapolating that value
    to a 60 dB decay. The formula used is:

        EDT = 6 * (time to decay from 0 dB to -10 dB)

    Steps:
    1. Convert multi-channel audio to mono if necessary.
    2. Compute the squared signal and then the normalized EDC.
    3. Convert the EDC to dB scale.
    4. Find the first time index where the EDC falls to -10 dB.
    5. Extrapolate to obtain the EDT.
    """
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    squared_signal = audio_data**2
    edc = np.flip(np.cumsum(np.flip(squared_signal)))
    edc = edc / np.max(edc)
    edc_db = 10 * np.log10(edc + 1e-10)

    # Find the first index where the decay reaches -10 dB
    idx_10 = np.where(edc_db <= -10)[0]
    if len(idx_10) == 0:
        return None
    else:
        idx_10 = idx_10[0]
        time_10 = idx_10 / sample_rate  # Time in seconds at which -10 dB is reached
        edt = 6 * time_10  # Extrapolate to a 60 dB decay (60/10 = 6)
        return edt


def generate_impulse_response_plot(audio_data, sample_rate):
    """
    Generate a time-domain plot of the Room Impulse Response (RIR).

    The plot displays amplitude vs. time and is saved into an in-memory buffer.
    The image is then encoded in base64 so it can be easily embedded into HTML.
    """
    times = np.arange(len(audio_data)) / sample_rate  # Create time axis
    plt.figure(figsize=(8, 4))
    plt.plot(times, audio_data, color="#3498db")
    plt.title("Impulse Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    # Save the figure to a buffer in PNG format
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()  # Close the figure to free memory
    return img_data


def generate_edc_plot(audio_data, sample_rate):
    """
    Generate a plot of the Energy Decay Curve (EDC) in decibels.

    The function computes the normalized cumulative energy decay and then plots it.
    The resulting plot is saved to a buffer and encoded in base64 for web embedding.
    """
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    squared_signal = audio_data**2
    edc = np.flip(np.cumsum(np.flip(squared_signal)))
    edc = edc / np.max(edc)
    edc_db = 10 * np.log10(edc + 1e-10)
    times = np.arange(len(edc_db)) / sample_rate
    plt.figure(figsize=(8, 4))
    plt.plot(times, edc_db, color="#e74c3c")
    plt.title("Energy Decay Curve")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (dB)")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return img_data


def generate_frequency_response_plot(audio_data, sample_rate):
    """
    Generate a frequency response plot from the audio data.

    The function computes the Fast Fourier Transform (FFT) of the audio data,
    converts the magnitude to decibels, and plots the frequency response on a logarithmic x-axis.
    The resulting plot is saved to a buffer and encoded in base64.
    """
    fft_data = np.abs(np.fft.rfft(audio_data))
    freqs = np.fft.rfftfreq(len(audio_data), d=1.0 / sample_rate)
    plt.figure(figsize=(8, 4))
    plt.semilogx(freqs, 20 * np.log10(fft_data + 1e-10), color="#2ecc71")
    plt.title("Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return img_data


def analyze_rir(file_path):
    """
    Process the uploaded Room Impulse Response (RIR) file and compute various acoustic parameters.

    Steps:
    1. Load the audio file using librosa, preserving the original sample rate and converting to mono.
    2. Compute acoustic metrics: RT60, Clarity (C80), Clarity (C60), and Early Decay Time (EDT).
    3. Determine the duration of the audio.
    4. Generate visualizations: impulse response, energy decay curve, and frequency response.
    5. Provide recommendations based on computed values.
    6. Return all computed metrics and plots in a dictionary.
    """
    # Load audio file (ensuring original sample rate and mono conversion)
    audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)

    # Compute acoustic metrics
    rt60 = calculate_rt60(audio_data, sample_rate)
    clarity = calculate_clarity(audio_data, sample_rate)  # Typically for C80
    c60 = calculate_c60(audio_data, sample_rate)  # Clarity index for speech (C60)
    edt = calculate_edt(audio_data, sample_rate)  # Early Decay Time
    duration = len(audio_data) / sample_rate  # Total duration in seconds

    # Generate visualizations
    impulse_plot = generate_impulse_response_plot(audio_data, sample_rate)
    edc_plot = generate_edc_plot(audio_data, sample_rate)
    frequency_plot = generate_frequency_response_plot(audio_data, sample_rate)

    # Provide basic recommendations based on the analysis
    recommendations = []
    if rt60 is not None and rt60 > 1.2:
        recommendations.append(
            "Reverberation time is high. Consider adding sound absorbers."
        )
    if clarity < 0:
        recommendations.append(
            "Clarity (C80) is low. Consider optimizing room dimensions or adding diffusers."
        )

    # Package all results into a dictionary
    results = {
        "rt60": rt60 if rt60 is not None else "N/A",
        "edt": edt if edt is not None else "N/A",  # Early Decay Time result
        "clarity": clarity,
        "c60": c60,
        "duration": duration,
        "sample_rate": sample_rate,
        "impulse_plot": impulse_plot,
        "edc_plot": edc_plot,
        "frequency_plot": frequency_plot,
        "recommendations": recommendations,
    }

    return results


@app.route("/")
def index():
    """
    Render the main page with the file upload form.
    """
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Endpoint to handle the file upload, process the RIR file, and return acoustic analysis results in JSON format.
    """
    # Check if a file is part of the request
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Verify that the file has a valid extension
    if file and allowed_file(file.filename):
        # Ensure the upload folder exists
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        try:
            # Analyze the uploaded RIR file
            results = analyze_rir(filepath)
            return jsonify(results)
        except Exception as e:
            # Return any exceptions that occur during processing
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up the file after processing
            if os.path.exists(filepath):
                os.remove(filepath)

    # Return an error if the file type is invalid
    return jsonify({"error": "Invalid file type"}), 400


if __name__ == "__main__":
    # Run the Flask development server with debug mode enabled for easier troubleshooting
    app.run(debug=True)
