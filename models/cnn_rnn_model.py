import torch
import torch.nn as nn

class SpeakerCNN_RNN(nn.Module):
    def __init__(self, num_speakers, num_mfcc=40, hidden_dim=128, rnn_layers=2, dropout_rate=0.3):
        super(SpeakerCNN_RNN, self).__init__()

        # CNN layers for feature extraction from MFCCs
        # Input: (batch_size, num_mfcc, sequence_length_frames)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_mfcc, 64, kernel_size=5, padding=2), # Output: (B, 64, L)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Output: (B, 64, L/2)

            nn.Conv1d(64, 128, kernel_size=5, padding=2), # Output: (B, 128, L/2)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Output: (B, 128, L/4)

            nn.Conv1d(128, 256, kernel_size=5, padding=2), # Output: (B, 256, L/4)
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # Output: (B, 256, L/8)
        )

        # The output of CNN layers will be (batch_size, features, new_sequence_length)
        # We need to calculate the actual sequence length after pooling to correctly initialize RNN
        # Let's assume input sequence length for MFCCs of 30s @ 16kHz, hop_length=160
        # Frame length = (16000 * 30) = 480000 samples
        # Number of frames = (480000 - 400) / 160 + 1 = ~3000 frames
        # After 3 MaxPool1d(kernel_size=2), sequence length becomes 3000 / 2 / 2 / 2 = 375 frames

        rnn_input_size = 256 # Number of features from CNN output

        # RNN (GRU) layers for temporal modeling
        # Input to RNN: (batch_size, sequence_length_frames, features)
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            bidirectional=True, # Bidirectional GRU for better context over time
            batch_first=True # Input and output tensors are provided as (batch, seq, feature)
        )

        # Global Average Pooling after RNN to get a fixed-size embedding for classification
        # We take the mean across the sequence dimension (dim=1)
        # Input for global_pool needs to be (batch_size, features, sequence_length)
        # So we permute rnn_out back
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final fully connected layer for classification
        # hidden_dim * 2 because of bidirectional GRU
        self.fc_layer = nn.Linear(hidden_dim * 2, num_speakers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, num_mfcc, sequence_length_frames)

        # CNN layers
        x = self.conv_layers(x) # Output shape: (batch_size, 256, reduced_sequence_length_frames)

        # Permute for RNN input: (batch_size, sequence_length_frames, features)
        x = x.permute(0, 2, 1)

        # RNN layers
        rnn_out, _ = self.rnn(x) # rnn_out shape: (batch_size, sequence_length_frames, hidden_dim * 2)

        # Apply Global Average Pooling across the sequence length dimension (dim=1)
        # Squeeze the resulting 1-dimensional output
        # Input for global_pool needs to be (batch_size, features, sequence_length)
        # So we permute rnn_out back
        pooled_output = self.global_pool(rnn_out.permute(0, 2, 1)).squeeze(-1) # Output: (batch_size, hidden_dim * 2)

        # Dropout for regularization
        x = self.dropout(pooled_output)

        # Final fully connected layer for classification
        x = self.fc_layer(x)
        return x