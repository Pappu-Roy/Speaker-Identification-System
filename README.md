# Voice Biometrics Platform for Speaker Identification (Local Deployment)

## Project Overview

This repository contains the source code for a **Voice Biometrics Platform for Speaker Identification**, developed as a final year research project. This platform is designed for **local deployment**, allowing users to enroll their voice samples and identify individuals based on their spoken voice, all running on a single machine without external cloud services.

The project focuses on building a complete end-to-end system, from custom deep learning model training to a local web application for user interaction.

## Features

* **User Authentication:** Local user registration and login (data stored locally).
* **User Profile Management:** Users can provide and view personal information (Name, Gender, Age, Institution, Department/Class, Language), stored locally.
* **Voice Enrollment:** Authenticated users can record and submit their voice samples. These samples are processed to extract unique speaker embeddings, which are then stored in a **local database** for future identification.
* **Speaker Detection:** Users can record a new voice sample, and the system will attempt to identify the speaker by comparing the new voice's embedding against the locally enrolled embeddings.
* **Deep Learning Backend:** Utilizes custom-trained Convolutional Neural Network (CNN) or Recurrent Neural Network (RNN) models (trained from scratch) for robust speaker embedding extraction.
* **MFCC Feature Extraction:** Employs Mel-Frequency Cepstral Coefficients (MFCCs) as audio features, a standard in speech processing.

## Technologies Used

* **Frontend:**
    * HTML, CSS (Tailwind CSS for styling)
    * JavaScript (ES6+)
    * Fetch API (for communication with local Python backend)
* **Backend (Local Server):**
    * Python 3.9+ (e.g., using Flask or a simple HTTP server)
    * PyTorch (for Deep Learning model inference)
    * TorchAudio (for MFCC feature extraction and audio processing)
    * NumPy, SciPy, Pandas, Joblib (for data handling and model utilities)
* **Local Database/Storage:**
    * SQLite (for user profiles and speaker embeddings) OR
    * Local JSON/Pickle files (for simpler data persistence)
* **Deep Learning Models:**
    * Custom-built CNN (Convolutional Neural Network)
    * Custom-built RNN (Recurrent Neural Network - GRU)
* **Development Environment:** Google Colab (for model training), Local machine (for web app and local backend development)
* **Version Control:** Git, GitHub

## Project Structure

## Setup and Running the Project (Local Deployment)

To set up and run this project locally, you will train your deep learning model, configure a local Python backend, and serve the web application.

### 1. Deep Learning Model Training (Google Colab)

Your deep learning model needs to be trained and its weights saved.

1.  **Upload Data to Google Drive:**
    * Ensure `training.csv` and the `voices/` folder (containing all your audio files) are uploaded to your Google Drive, e.g., in `My Drive/project/`.
2.  **Open Colab Notebook:**
    * Open either `CNN_Model_Final.ipynb` (for CNN only) or `RNN_Model_Final.ipynb` (for RNN only) or `CNN_RNN_Model.ipynb` (for CNN-RNN combined) in Google Colab.
    * **Crucial:** Go to `Runtime` -> `Change runtime type` -> select `GPU`.
3.  **Run All Cells:** Execute all cells in the notebook. This will:
    * Install necessary libraries.
    * Mount your Google Drive.
    * Load and preprocess your audio data (MFCC extraction, padding/truncation).
    * Train your chosen CNN or RNN model from scratch.
    * Save the trained model weights (`speaker_cnn_best.pth` or `speaker_rnn_best.pth`) and the `speaker_mapping.joblib` file to your Google Drive (e.g., `My Drive/project/saved_models_cnn_only/` or `My Drive/project/saved_models_rnn_only/`).
4.  **Download Trained Models:**
    * After training, download your saved model files (`.pth` and `.joblib`) from your Google Drive to your local project's `models/` directory.

### 2. Local Python Backend Setup (e.g., Flask)

You will need a Python script (e.g., `app.py`) that acts as your local server, loads your trained model, and handles API requests from your web frontend.

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```
2.  **Install Python Dependencies:**
    * Create a `requirements.txt` file in your project root with:
        ```
        Flask
        torch==2.0.1
        torchaudio==2.0.2
        numpy==2.0.2
        scipy==1.15.3
        pandas==2.2.2
        scikit-learn==1.6.1
        joblib==1.5.1
        tqdm==4.67.1 # Not strictly needed for runtime, but good for consistency
        # Add any other packages your app.py or model loading needs
        ```
    * Install: `pip install -r requirements.txt`
3.  **Implement `app.py`:**
    * This file will contain your Flask application.
    * It will load your trained `.pth` model and `.joblib` mapping.
    * It will define API endpoints for:
        * `/register`: To handle user sign-up and store profile data (e.g., in SQLite).
        * `/login`: To authenticate users locally.
        * `/enroll`: To receive audio, extract embedding, and store it locally (e.g., in SQLite or JSON).
        * `/detect`: To receive audio, extract embedding, compare it to stored embeddings, and return the detected speaker.
    * **Note:** Implementing user authentication and a local database (like SQLite) from scratch requires additional Python code.

### 3. Web Application (Frontend)

The web application (in `templates/index.html` and `static/js/main.js`) will communicate with your local Python backend.

1.  **Update Frontend JavaScript:**
    * Modify your JavaScript in `static/js/main.js` (or directly in `index.html`) to make `fetch` requests to your local Python backend's endpoints (e.g., `http://localhost:5000/enroll`, `http://localhost:5000/detect`) instead of Firebase Cloud Functions.
    * Remove all Firebase SDK imports and related logic from the frontend. Implement local authentication and data storage within your Flask app.

### 4. Running the Local Project

1.  **Start the Local Python Backend:**
    * Open your terminal/Git Bash in the project root.
    * Activate your virtual environment: `source venv/bin/activate` (Windows: `.\venv\Scripts\activate`)
    * Run your Flask app (e.g., `python app.py`). This will typically start a server on `http://localhost:5000`.
2.  **Open the Web Application:**
    * Open `templates/index.html` directly in your web browser. (Some browsers might have CORS restrictions for local file access; if so, serve it via `python -m http.server 8000` from the `templates` directory, or configure Flask to serve static files).
    * Interact with the web interface to register, log in, enroll voices, and detect speakers.

## Performance Considerations

* **Dataset Size:** The model is trained from scratch on a relatively small dataset (147 audio files, 52 speakers). This will severely limit its generalization capabilities. Expect lower accuracy and potential overfitting, especially on voices not closely resembling the training data.
* **Hardware:** The entire system (web app, Python backend, and model inference) will run on your local machine. Performance will depend heavily on your CPU and RAM.
* **Threshold Tuning:** The similarity threshold for speaker detection will need to be tuned based on your model's actual performance.

## Future Work

* **Larger Dataset:** Train on a significantly larger and more diverse dataset for improved accuracy and generalization.
* **Advanced Architectures:** Experiment with more sophisticated speaker recognition models.
* **Data Augmentation:** Implement audio augmentation techniques during training.
* **Multi-Speaker Scenarios:** Integrate speaker diarization and speech separation techniques.
* **Real-time Processing:** Optimize models for true real-time inference.
* **User Interface Enhancements:** Improve the web application's UI/UX.
* **Deployment:** Explore cloud deployment options (e.g., Docker, AWS, Google Cloud) for scalability and accessibility.

## Contributors

This Reaserch project was developed by:

* **Pappu Roy** ([https://github.com/Pappu-Roy])
* **Md. Sazim Mahmudur Rahman** ([https://github.com/Sazim2019331087])

## Contact

For any questions or inquiries, please contact:
Pappu Roy
[pappuroy10102000@gmail.com]
[https://github.com/Pappu-Roy]