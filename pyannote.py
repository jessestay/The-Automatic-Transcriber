import os
import sys
import subprocess
import torch
from whisper_timestamped import load_model, transcribe
from pyannote.audio import Pipeline
from datetime import timedelta
from dotenv import load_dotenv

def load_config():
    """Load configuration from .env file"""
    # Try to load from .env file
    if os.path.exists('.env'):
        load_dotenv()
    
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("❌ Error: HUGGINGFACE_TOKEN not found in .env file")
        print("Please create a .env file with your token like this:")
        print("HUGGINGFACE_TOKEN=your_token_here")
        sys.exit(1)
    
    return token

def convert_to_wav(input_file):
    output_file = os.path.splitext(input_file)[0] + ".wav"
    
    print(f"🔄 Converting {input_file} to {output_file}...")
    
    command = [
        "ffmpeg", "-i", input_file,
        "-ac", "1",                # Mono audio
        "-ar", "16000",            # 16kHz sampling rate
        "-af", "highpass=f=200,lowpass=f=3000",  # Filter out noise
        "-filter:a", "volume=1.5",  # Increase volume if audio is too quiet
        output_file,
        "-y"
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print("❌ Error converting file:", result.stderr.decode())
        sys.exit(1)
    
    print("✅ Conversion complete!")
    return output_file

def is_video_file(filename):
    video_extensions = {".mp4", ".mov", ".mkv", ".avi"}
    return os.path.splitext(filename)[1].lower() in video_extensions

def get_speaker_at_time(time, diarization):
    """Find which speaker was talking at a given time"""
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= time <= turn.end:
            return speaker
    return "Unknown Speaker"

def main():
    if len(sys.argv) < 2:
        print("❌ Error: Please provide a filename as an argument.")
        print("Usage: python transcribe.py <filename> [model_size]")
        sys.exit(1)

    # Load token from config
    HUGGINGFACE_TOKEN = load_config()

    input_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"

    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' not found.")
        sys.exit(1)

    # Convert video files to audio
    if is_video_file(input_file):
        input_file = convert_to_wav(input_file)

    print("🎙️ Running speaker diarization...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HUGGINGFACE_TOKEN
    )
    
    # Move to CPU or GPU after loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    
    # Add parameters to improve diarization
    diarization = pipeline(input_file, 
                          min_speakers=2,          # Minimum number of speakers
                          max_speakers=3)          # Maximum number of speakers

    print(f"🚀 Running transcription using {model_size} model...")
    model = load_model(model_size)
    # Add language and initial prompt to improve transcription
    result = transcribe(model, input_file, 
                       language="en",     # Specify language if known
                       condition_on_previous_text=True,  # Use context from previous segments
                       beam_size=5)       # Increase beam size for better accuracy

    # Create output filename
    output_file = os.path.splitext(input_file)[0] + "_transcript.txt"

    # Write results to file and print to console
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in result['segments']:
            start_time = segment['start']
            end_time = segment['end']
            speaker = get_speaker_at_time(start_time, diarization)
            timestamp = f"[{start_time:.1f}s - {end_time:.1f}s]"
            text = segment['text'].strip()
            line = f"{timestamp} {speaker}: {text}"
            print(line)
            f.write(line + "\n")

    print(f"✅ Transcription with speaker diarization complete! Saved to {output_file}")

if __name__ == "__main__":
    main()