import torch
import torchaudio
from torchaudio.transforms import MFCC
from torch.utils.data import Dataset, DataLoader # Included for class definition context

class SpeakerDatasetMFCC(Dataset):
    def __init__(self, dataframe, target_sr=16000, num_mfcc=40, n_fft=400, hop_length=160):
        self.dataframe = dataframe
        self.target_sr = target_sr
        self.num_mfcc = num_mfcc
        self.max_len_sec = 30 # Fixed duration for training
        self.max_len_samples = self.max_len_sec * self.target_sr

        self.mfcc_transform = MFCC(
            sample_rate=target_sr, n_mfcc=num_mfcc, melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'n_mels': 128}
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path = row['audio_path']
        label = row['speaker_id']

        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1: # Convert to mono if stereo
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            # Ensure waveform is 2D (channels, samples) expected by MFCC transform
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0) # Add channel dimension if it's just (samples,)

            # Pad or truncate to a fixed length (max_len_samples)
            if waveform.shape[1] > self.max_len_samples:
                waveform = waveform[:, :self.max_len_samples]
            elif waveform.shape[1] < self.max_len_samples:
                padding = self.max_len_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            mfcc_features = self.mfcc_transform(waveform)

            # --- CRITICAL FIX: Squeeze the channel dimension (dim=0) here ---
            # mfcc_features original shape is (1, num_mfcc, num_frames) for mono audio
            # We want (num_mfcc, num_frames) for the Conv1d input after batching
            if mfcc_features.ndim == 3 and mfcc_features.shape[0] == 1:
                mfcc_features = mfcc_features.squeeze(0)

            return mfcc_features, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            # In a deployed setting, you might log this error rather than printing
            # and decide how to handle failed samples (e.g., return None as done here)
            print(f"Error processing {audio_path}: {e}. Skipping this sample.")
            return None, None

# Custom collate_fn to handle None values (from failed audio loads) and ensure consistent stacking
def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None] # Filter out samples that returned None
    if not batch:
        return None, None # Return None if the batch is empty after filtering

    mfccs, labels = zip(*batch)

    # All MFCCs in `mfccs` should now have the same (num_mfcc, num_frames) shape
    mfccs_stacked = torch.stack(mfccs)
    labels_stacked = torch.stack(labels)

    return mfccs_stacked, labels_stacked