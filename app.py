import os
import torch
import torchaudio
from torchaudio.transforms import MFCC
import joblib
import io
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import sqlite3
import uuid # For generating unique filenames for temp files
import subprocess # To run ffmpeg command

# Import your model definition and dataset logic from local files
from models.cnn_rnn_model import SpeakerCNN_RNN

torchaudio.set_audio_backend("soundfile") # Ensure soundfile is the backend

app = Flask(__name__)
app.secret_key = os.urandom(24) # Replace with a strong, random key in production
app.permanent_session_lifetime = timedelta(minutes=30)

# --- Database Setup (Local SQLite) ---
DATABASE = 'instance/app.db' # SQLite database file

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    return conn

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                name TEXT,
                age INTEGER,
                gender TEXT,
                institution TEXT,
                class_dept TEXT,
                language TEXT
            )
        ''')
        db.commit()
        db.close()

# --- ML Model Loading (Global for efficiency) ---
MODEL_PATH = 'models/speaker_cnn_rnn_best.pth'
SPEAKER_MAPPING_PATH = 'models/speaker_mapping.joblib'

model = None
speaker_mapping = None
mfcc_transform_inference = None
id_to_email_mapping = {} # Initialize globally

# ML model parameters (must match your training)
NUM_MFCC = 40
HIDDEN_DIM = 128
RNN_LAYERS = 2
DROPOUT_RATE = 0.3
SAMPLE_RATE = 16000 # This is your target sample rate for the model
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
MAX_LEN_SECONDS = 30 # Max audio length for inference

def load_ml_assets():
    global model, speaker_mapping, mfcc_transform_inference, id_to_email_mapping
    try:
        # Load speaker mapping
        speaker_mapping = joblib.load(SPEAKER_MAPPING_PATH)
        # Reverse mapping for prediction output
        id_to_email_mapping = speaker_mapping # Assign directly, as it's already ID -> Email


        num_speakers = len(speaker_mapping)

        # Initialize the model architecture
        model = SpeakerCNN_RNN(num_speakers=num_speakers, num_mfcc=NUM_MFCC,
                               hidden_dim=HIDDEN_DIM, rnn_layers=RNN_LAYERS,
                               dropout_rate=DROPOUT_RATE)
        
        # Load the trained weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))) 
        model.eval() # Set to evaluation mode

        # Initialize MFCC transform for inference
        mfcc_transform_inference = MFCC(
            sample_rate=SAMPLE_RATE, n_mfcc=NUM_MFCC, 
            melkwargs={'n_fft': N_FFT, 'hop_length': HOP_LENGTH, 'n_mels': N_MELS}
        )
        print("ML Model and assets loaded successfully!")
    except Exception as e:
        print(f"Error loading ML assets: {e}")
        model = None
        speaker_mapping = None
        mfcc_transform_inference = None
        id_to_email_mapping = {} # Reset in case of failure

# Initialize database and load ML assets on app startup
with app.app_context():
    init_db()
    load_ml_assets()

# --- Frontend Routes (Serving HTML pages) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/profile')
def profile():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('profile.html')

@app.route('/detect')
def detect():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('detect.html')


# --- Backend API Routes ---

# ... (rest of imports and initial Flask setup)

# global speaker_mapping and id_to_email_mapping are already loaded at startup
# (from your load_ml_assets() function)

# ... (rest of frontend routes and existing API routes)

@app.route('/api/check_speaker_status', methods=['GET'])
def api_check_speaker_status():
    if 'user_email' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_email = request.args.get('email')
    if not user_email:
        return jsonify({"error": "Email parameter is missing"}), 400

    if not speaker_mapping:
        return jsonify({"error": "Speaker mapping not loaded."}), 500

    # Check if the user's email exists as a value in the speaker_mapping (ID -> email)
    # The 'speaker_mapping' variable itself is in the format {ID: 'email'}
    # So we need to check if the user_email is present in the VALUES of the mapping
    is_enrolled = user_email in speaker_mapping.values()

    print(f"Checking enrollment for {user_email}: Is Enrolled = {is_enrolled}")
    return jsonify({"status": "success", "is_enrolled": is_enrolled}), 200

# ... (rest of your app.py code)

@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    age = data.get('age')
    gender = data.get('gender')
    institution = data.get('institution')
    class_dept = data.get('class_dept')
    language = data.get('language')

    if not all([email, password, name, age, gender]):
        return jsonify({"error": "Missing required fields"}), 400

    hashed_password = generate_password_hash(password) # Removed 'method' argument
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (email, password, name, age, gender, institution, class_dept, language) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (email, hashed_password, name, age, gender, institution, class_dept, language))
        conn.commit()
        return jsonify({"status": "success", "message": "Account created successfully!"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered."}), 409
    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({"error": "An internal error occurred during signup."}), 500
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if user and check_password_hash(user['password'], password):
        session['user_email'] = user['email'] # Store email in session
        session.permanent = True
        return jsonify({"status": "success", "message": "Logged in successfully!"}), 200
    else:
        return jsonify({"error": "Invalid email or password."}), 401

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.pop('user_email', None)
    return jsonify({"status": "success", "message": "Logged out successfully!"}), 200

@app.route('/api/profile_data', methods=['GET'])
def api_profile_data():
    user_email = request.args.get('email')
    if 'user_email' not in session or session['user_email'] != user_email:
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (user_email,)).fetchone()
    conn.close()
    
    if user:
        # Convert sqlite3.Row to dictionary for JSON serialization
        user_dict = dict(user)
        # Remove password hash for security
        user_dict.pop('password', None)
        return jsonify({"status": "success", "user": user_dict}), 200
    return jsonify({"error": "User not found."}), 404

@app.route('/api/detect_speaker', methods=['POST'])
def api_detect_speaker():
    if 'user_email' not in session: # Check if logged in locally
        return jsonify({"error": "Unauthorized. Please log in."}), 401
    
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not model or not mfcc_transform_inference or not speaker_mapping:
        print("ML Model not loaded at inference time. Restart server if this persists.")
        return jsonify({"error": "ML model not ready. Please try again later."}), 503

    temp_audio_dir = 'temp_audio_uploads'
    os.makedirs(temp_audio_dir, exist_ok=True)
    
    original_temp_file_path = os.path.join(temp_audio_dir, f"{uuid.uuid4()}_original.{audio_file.filename.split('.')[-1]}")
    converted_temp_file_path = os.path.join(temp_audio_dir, f"{uuid.uuid4()}_converted.wav") # Target WAV format

    try:
        audio_file.save(original_temp_file_path) # Save the uploaded file temporarily

        # --- NEW FFmpeg CONVERSION STEP ---
        # Convert to 16kHz, 16-bit PCM, mono WAV to ensure compatibility
        # -i: input file, -ar: audio sample rate, -ac: audio channels, -acodec: audio codec, -y: overwrite output
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', original_temp_file_path,
            '-ar', str(SAMPLE_RATE),  # Target sample rate (16000 Hz)
            '-ac', '1',               # Mono channel
            '-acodec', 'pcm_s16le',   # 16-bit signed PCM, little-endian
            '-y', converted_temp_file_path
        ]
        
        # Execute FFmpeg command
        print(f"Running FFmpeg conversion: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"FFmpeg conversion successful to {converted_temp_file_path}")

        # Load audio from the CONVERTED temporary file path
        waveform, sample_rate = torchaudio.load(converted_temp_file_path)
        
        # Rest of your preprocessing logic remains the same (resampling/mono/padding should largely be handled by ffmpeg now)
        # Still keep these for robustness against imperfect ffmpeg conversion or non-standard inputs
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1: # Convert to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.ndim == 1: # Ensure 2D (channels, samples)
            waveform = waveform.unsqueeze(0)

        # Pad or truncate to fixed length (30 seconds)
        max_len_samples = MAX_LEN_SECONDS * SAMPLE_RATE
        if waveform.shape[1] > max_len_samples:
            waveform = waveform[:, :max_len_samples]
        elif waveform.shape[1] < max_len_samples:
            padding = max_len_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Extract MFCC features
        mfcc_features = mfcc_transform_inference(waveform)
        if mfcc_features.ndim == 3 and mfcc_features.shape[0] == 1:
            mfcc_features = mfcc_features.squeeze(0)
        
        # Add batch dimension for the model (expected input: Batch x Num_MFCC x Frames)
        mfcc_features = mfcc_features.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            outputs = model(mfcc_features)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_speaker_id = predicted_idx.item()

            
            predicted_email = id_to_email_mapping.get(predicted_speaker_id, "Unknown Speaker")

        return jsonify({"status": "success", "predicted_speaker": predicted_email}), 200

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed: {e.stderr}")
        return jsonify({"error": f"Audio format conversion failed. FFmpeg Error: {e.stderr}"}), 500
    except Exception as e:
        print(f"Error during audio processing or inference: {e}")
        return jsonify({"error": f"Failed to process audio for detection: {e}"}), 500
    finally:
        # Clean up both original and converted temporary files
        if os.path.exists(original_temp_file_path):
            os.remove(original_temp_file_path)
        if os.path.exists(converted_temp_file_path):
            os.remove(converted_temp_file_path)


@app.route('/api/enroll_voice', methods=['POST'])
def api_enroll_voice():
    if 'user_email' not in session: # Check if logged in locally
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    if 'audio_samples' not in request.files:
        return jsonify({"error": "No audio samples provided"}), 400
    
    audio_samples = request.files.getlist('audio_samples')
    if not audio_samples:
        return jsonify({"error": "No audio files selected for enrollment"}), 400

    if not model or not mfcc_transform_inference:
        print("ML Model not loaded for enrollment. Restart server if this persists.")
        return jsonify({"error": "ML model not ready for enrollment. Please try again later."}), 503

    processed_features = []
    current_user_email = session['user_email']
    
    temp_audio_dir = 'temp_audio_uploads'
    os.makedirs(temp_audio_dir, exist_ok=True)
    
    try:
        for audio_file in audio_samples:
            original_temp_file_path = os.path.join(temp_audio_dir, f"{uuid.uuid4()}_original.{audio_file.filename.split('.')[-1]}")
            converted_temp_file_path = os.path.join(temp_audio_dir, f"{uuid.uuid4()}_converted.wav")

            audio_file.save(original_temp_file_path) # Save the uploaded file temporarily

            # --- NEW FFmpeg CONVERSION STEP (inside loop) ---
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', original_temp_file_path,
                '-ar', str(SAMPLE_RATE),  # Target sample rate (16000 Hz)
                '-ac', '1',               # Mono channel
                '-acodec', 'pcm_s16le',   # 16-bit signed PCM, little-endian
                '-y', converted_temp_file_path
            ]
            print(f"Running FFmpeg conversion for enrollment: {' '.join(ffmpeg_cmd)}")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            print(f"FFmpeg conversion successful to {converted_temp_file_path}")
            # --- END NEW FFmpeg CONVERSION STEP ---

            waveform, sample_rate = torchaudio.load(converted_temp_file_path) # Load from CONVERTED temporary file

            # Rest of your preprocessing logic remains the same
            if sample_rate != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            # Pad or truncate
            max_len_samples = MAX_LEN_SECONDS * SAMPLE_RATE
            if waveform.shape[1] > max_len_samples:
                waveform = waveform[:, :max_len_samples]
            elif waveform.shape[1] < max_len_samples:
                padding = max_len_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            mfcc_features = mfcc_transform_inference(waveform).squeeze(0)
            processed_features.append(mfcc_features.numpy()) 
            
            # Clean up individual temp files after processing
            if os.path.exists(original_temp_file_path):
                os.remove(original_temp_file_path)
            if os.path.exists(converted_temp_file_path):
                os.remove(converted_temp_file_path)

        print(f"User {current_user_email} enrolled {len(processed_features)} audio samples.")
        return jsonify({"status": "success", "message": f"Successfully processed {len(processed_features)} audio samples for enrollment for user {current_user_email}. In a real system, these would be stored and used for future recognition."}), 200

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed: {e.stderr}")
        return jsonify({"error": f"Audio format conversion failed. FFmpeg Error: {e.stderr}"}), 500
    except Exception as e:
        print(f"Error during audio processing or enrollment: {e}")
        return jsonify({"error": f"Failed to process audio for enrollment: {e}"}), 500
    # No 'finally' here for collective cleanup as files are removed individually


if __name__ == '__main__':
    os.makedirs('instance', exist_ok=True)
    os.makedirs('temp_audio_uploads', exist_ok=True) # Create temp dir for uploads
    app.run(debug=True, host='127.0.0.1', port=5000)