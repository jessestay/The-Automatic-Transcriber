import os
import sys
import subprocess
import torch
from whisper_timestamped import load_model, transcribe
from pyannote.audio import Pipeline
from datetime import timedelta
from dotenv import load_dotenv
import shutil
import platform
import argparse

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

def get_ffmpeg_path():
    # First try to find ffmpeg in PATH
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
        
    # If not in PATH, check common locations based on OS
    if platform.system() == 'Windows':
        common_locations = [
            r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe"
        ]
    else:  # Linux/Mac
        common_locations = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg"
        ]
    
    for location in common_locations:
        if os.path.isfile(location):
            return location
            
    raise FileNotFoundError("FFmpeg not found. Please install FFmpeg and make sure it's in your PATH.")

def convert_to_wav(input_file):
    output_file = os.path.splitext(input_file)[0] + '.wav'
    print(f"🔄 Converting {input_file} to {output_file}...")
    
    ffmpeg_path = get_ffmpeg_path()
    command = [ffmpeg_path, '-i', input_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_file]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("Error converting file:")
        print(result.stderr.decode())
        sys.exit(1)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input video/audio file")
    parser.add_argument("--model", default="large-v2", help="Whisper model size (tiny, base, small, medium, large-v2)")
    parser.add_argument("--language", default="en", help="Language code (en, fr, etc.)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load token from config
    HUGGINGFACE_TOKEN = load_config()

    input_file = args.input_file
    model_size = args.model
    language = args.language
    device = args.device

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
    pipeline = pipeline.to(device)
    
    # Add parameters to improve diarization
    diarization = pipeline(input_file, 
                          min_speakers=2,          # Minimum number of speakers
                          max_speakers=3)          # Maximum number of speakers

    print(f"🚀 Running transcription using {model_size} model...")
    model = load_model(model_size)
    # Add language and initial prompt to improve transcription
    result = transcribe(model, input_file, 
                       language=language,     # Specify language if known
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