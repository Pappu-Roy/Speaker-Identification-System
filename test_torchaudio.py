import torchaudio
import os
import torch # Needed for tensor operations like .unsqueeze()
torchaudio.set_audio_backend("soundfile")

# Path to the simple WAV file you will use for testing
# Make sure you place a simple WAV file named 'test_audio.wav'
# in the same directory as this script.
TEST_AUDIO_PATH = 'test_audio.wav' 

def test_load_audio():
    if not os.path.exists(TEST_AUDIO_PATH):
        print(f"Error: Test audio file not found at {TEST_AUDIO_PATH}")
        print("Please create a simple WAV file (e.g., 16bit, 16kHz, mono) and save it as 'test_audio.wav' in the same directory as this script (voice-recognition-app-local/).")
        print("You can use Windows Sound Recorder or Audacity to create one.")
        return

    try:
        print(f"Attempting to load audio from: {TEST_AUDIO_PATH}")
        waveform, sample_rate = torchaudio.load(TEST_AUDIO_PATH)
        
        print(f"Successfully loaded audio from {TEST_AUDIO_PATH}")
        print(f"Waveform shape: {waveform.shape}")
        print(f"Sample rate: {sample_rate} Hz")

        # --- Begin MFCC processing (like your app.py) ---
        # These parameters must match your app.py and model training
        NUM_MFCC = 40
        TARGET_SAMPLE_RATE = 16000 # Your model's expected sample rate
        N_FFT = 400
        HOP_LENGTH = 160
        N_MELS = 128
        MAX_LEN_SECONDS = 30
        
        # Resample if necessary
        if sample_rate != TARGET_SAMPLE_RATE:
            print(f"Resampling from {sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz...")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = TARGET_SAMPLE_RATE # Update sample_rate after resampling

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            print("Converting stereo to mono...")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.ndim == 1: # Ensure 2D (channels, samples)
            waveform = waveform.unsqueeze(0)

        # Pad or truncate to fixed length
        max_len_samples = MAX_LEN_SECONDS * TARGET_SAMPLE_RATE
        if waveform.shape[1] > max_len_samples:
            print(f"Truncating audio from {waveform.shape[1]} samples to {max_len_samples} samples...")
            waveform = waveform[:, :max_len_samples]
        elif waveform.shape[1] < max_len_samples:
            padding = max_len_samples - waveform.shape[1]
            print(f"Padding audio from {waveform.shape[1]} samples to {max_len_samples} samples with {padding} zeros...")
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=TARGET_SAMPLE_RATE, n_mfcc=NUM_MFCC, 
            melkwargs={'n_fft': N_FFT, 'hop_length': HOP_LENGTH, 'n_mels': N_MELS}
        )
        
        mfcc_features = mfcc_transform(waveform)
        
        # Squeeze channel dim if it's there (e.g., from mono (1, N_mfcc, N_frames) to (N_mfcc, N_frames))
        if mfcc_features.ndim == 3 and mfcc_features.shape[0] == 1:
            mfcc_features = mfcc_features.squeeze(0)

        print(f"Successfully extracted MFCC features. Shape: {mfcc_features.shape}")
        # --- End MFCC processing ---

    except Exception as e:
        print(f"\nERROR: Failed to load or process audio in test script: {e}")
        print("This indicates a core issue with your torchaudio installation or its ability to process this specific audio file.")
        print("Possible causes:")
        print("  - torchaudio installation is corrupted/incomplete.")
        print("  - FFmpeg is not correctly linked (even if 'ffmpeg -version' works).")
        print("  - The 'test_audio.wav' file itself is malformed or uses an unsupported codec.")


if __name__ == "__main__":
    test_load_audio()