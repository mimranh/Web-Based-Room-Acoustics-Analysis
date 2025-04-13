These instructions are for Mr. Imran or any teacher in TU Ilmenau 
if they want to test this project in their local maschine. Please 
make sure you have python 3.8.19 installed in your machine.


1. Download the project folder to your local machine.

2. In a terminal or command prompt, navigate to the project folder:
   cd Accoustic_Analyzer_by_Waqar

3. Create and activate a virtual environment:
   - On Windows:
       python -m venv venv
       venv\Scripts\activate
   - On macOS/Linux:
       python3 -m venv venv
       source venv/bin/activate

4. Install the required dependencies:
    pip install Flask numpy scipy librosa matplotlib
    

5. Run the Flask application:
   python app.py

6. Open your web browser and go to:
   http://127.0.0.1:5000

That's it! You can now upload a .wav RIR file from audios folder and analyze the room acoustics.
