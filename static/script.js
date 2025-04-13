// Listen for the form submission event on the upload form
document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();  // Prevent default form submission behavior

    // Get references to key DOM elements
    const fileInput = document.getElementById('rir-file');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error-message');
    const loader = document.getElementById('loader');

    // Clear any previous error messages and hide results; show loader during processing
    errorDiv.classList.add('d-none');
    resultsDiv.classList.add('d-none');
    loader.classList.remove('d-none');

    // Ensure that a file is selected
    if (!fileInput.files.length) {
        showError('Please select a file.');
        loader.classList.add('d-none');
        return;
    }

    // Prepare the file data to be sent to the server
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        // Make the POST request to the backend analyze endpoint
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        // Throw an error if the response is not ok
        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed.');
        }

        // Update metric values in the DOM:
        // RT60: Reverberation time (in seconds)
        document.getElementById('rt60-value').textContent =
            typeof data.rt60 === 'number' ? data.rt60.toFixed(2) : data.rt60;

        // EDT: Early Decay Time (in seconds)
        document.getElementById('edt-value').textContent =
            typeof data.edt === 'number' ? data.edt.toFixed(2) : data.edt;

        // Clarity (C80): Clarity metric typically for music (in dB)
        document.getElementById('clarity-value').textContent = data.clarity.toFixed(2);

        // Clarity (C60): Clarity metric for speech (in dB)
        document.getElementById('c60-value').textContent = data.c60.toFixed(2);

        // Duration of the audio file (in seconds)
        document.getElementById('duration-value').textContent = data.duration.toFixed(2);

        // Sample Rate: Audio sampling rate (in Hz)
        document.getElementById('sample-rate-value').textContent = data.sample_rate;

        // Update visualization images (plots) using the base64 encoded data
        document.getElementById('impulse-plot').src = 'data:image/png;base64,' + data.impulse_plot;
        document.getElementById('edc-plot').src = 'data:image/png;base64,' + data.edc_plot;
        document.getElementById('frequency-plot').src = 'data:image/png;base64,' + data.frequency_plot;

        // Update recommendations if available
        const recDiv = document.getElementById('recommendations');
        recDiv.innerHTML = ''; // Clear any previous recommendations
        if (data.recommendations && data.recommendations.length > 0) {
            const recHeader = document.createElement('h4');
            recHeader.textContent = 'Recommendations:';
            recDiv.appendChild(recHeader);
            const recList = document.createElement('ul');
            data.recommendations.forEach(rec => {
                const li = document.createElement('li');
                li.textContent = rec;
                recList.appendChild(li);
            });
            recDiv.appendChild(recList);
        }

        // Finally, show the results section with updated metrics and visualizations
        resultsDiv.classList.remove('d-none');
    } catch (error) {
        // In case of an error, display an error message to the user
        showError(error.message);
    } finally {
        // Hide the loader regardless of success or failure
        loader.classList.add('d-none');
    }
});

/**
 * Display an error message in the UI.
 * @param {string} message - The error message to display.
 */
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.classList.remove('d-none');
    document.getElementById('results').classList.add('d-none');
}
