import yaml
import os
import look2hear.models
import argparse
import torch
import torchaudio
import torchaudio.transforms as T # Added for resampling

# audio path
parser = argparse.ArgumentParser()
# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Separate speech sources using Look2Hear TIGER model.")
parser.add_argument("--audio_path", default="test/mix.wav", help="Path to audio file (mixture).")
parser.add_argument("--output_dir", default="separated_audio", help="Directory to save separated audio files.")
parser.add_argument("--model_cache_dir", default="cache", help="Directory to cache downloaded model.")

# Parse arguments once at the beginning

args = parser.parse_args()

audio_path = args.audio_path

output_dir = args.output_dir

cache_dir = args.model_cache_dir
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load model

print("Loading TIGER model...")
# Ensure cache directory exists if specified
if cache_dir:
    os.makedirs(cache_dir, exist_ok=True)
# Load the pretrained model
model = look2hear.models.TIGER.from_pretrained("JusperLee/TIGER-speech", cache_dir=cache_dir)
model.to(device)
model.eval()


# --- Audio Loading and Preprocessing ---
# Define the target sample rate expected by the model (usually 16kHz for TIGER)

target_sr = 16000
print(f"Loading audio from: {audio_path}")
try:
    # Load audio and get its original sample rate
    waveform, original_sr = torchaudio.load(audio_path)
except Exception as e:
    print(f"Error loading audio file {audio_path}: {e}")
    exit(1)
print(f"Original sample rate: {original_sr} Hz, Target sample rate: {target_sr} Hz")

# Resample if necessary
if original_sr != target_sr:
    print(f"Resampling audio from {original_sr} Hz to {target_sr} Hz...")
    resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
    waveform = resampler(waveform)
    print("Resampling complete.")

# Move waveform to the target device
audio = waveform.to(device)

# Prepare the input tensor for the model
# Model likely expects a batch dimension [B, T] or [B, C, T]
# Assuming input is mono or model handles channels; add batch dim
# If audio has channel dim [C, T], keep it. If it's just [T], add channel dim first.

if audio.dim() == 1:
    audio = audio.unsqueeze(0) # Add channel dimension -> [1, T]
# Add batch dimension -> [1, C, T]
# The original audio[None] is equivalent to unsqueeze(0) on the batch dimension
audio_input = audio.unsqueeze(0).to(device)
print(f"Audio tensor prepared with shape: {audio_input.shape}")

# --- Speech Separation ---

# Create output directory if it doesn't exist

os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")
print("Performing separation...")

with torch.no_grad():
    # Pass the prepared input tensor to the model
    ests_speech = model(audio_input)  # Expected output shape: [B, num_spk, T]

# Process the estimated sources

# Remove the batch dimension -> [num_spk, T]

ests_speech = ests_speech.squeeze(0)

num_speakers = ests_speech.shape[0]

print(f"Separation complete. Detected {num_speakers} potential speakers.")



# --- Save Separated Audio ---

# Dynamically save all separated tracks

for i in range(num_speakers):
    output_filename = os.path.join(output_dir, f"spk{i+1}.wav")
    speaker_track = ests_speech[i].cpu() # Get the i-th speaker track and move to CPU
    print(f"Saving speaker {i+1} to {output_filename}")
    speaker_track = speaker_track.unsqueeze(0) if speaker_track.dim() == 1 else speaker_track
    try:
        torchaudio.save(
            output_filename,
            speaker_track, # Save the individual track
            target_sr      # Save with the target sample rate
        )
    except Exception as e:
        print(f"Error saving file {output_filename}: {e}")
