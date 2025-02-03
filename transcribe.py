import os
import sys
import subprocess
import torch
from whisper_timestamped import load_model, transcribe
from pyannote.audio import Pipeline
from datetime import timedelta
from dotenv import load_dotenv
import whisperx
import gc
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

def transcribe_with_whisperx(input_file, model_size, HUGGINGFACE_TOKEN, device):
    """Transcribe using WhisperX"""
    print(f"🚀 Running WhisperX transcription using {model_size} model...")
    
    # 1. Transcribe with original whisper
    model = whisperx.load_model(model_size, device)
    result = model.transcribe(input_file, language="en", batch_size=16)
    
    # 2. Load alignment model and align
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, input_file, device)
    
    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HUGGINGFACE_TOKEN, device=device)
    diarize_segments = diarize_model(input_file)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # Clean up memory
    del model, model_a, metadata, diarize_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

def transcribe_with_whisper_pyannote(input_file, model_size, diarization):
    """Transcribe using original Whisper + Pyannote"""
    print(f"🚀 Running Whisper+Pyannote transcription using {model_size} model...")
    model = load_model(model_size)
    result = transcribe(model, input_file, 
                       language="en",
                       condition_on_previous_text=True,
                       beam_size=5)
    
    return result, diarization

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Transcribe audio with speaker diarization')
    parser.add_argument('input_file', help='Audio or video file to transcribe')
    parser.add_argument('--model-size', default='base', help='Model size (tiny, base, small, medium, large)')
    parser.add_argument('--use-whisperx', action='store_true', help='Use WhisperX instead of Whisper+Pyannote')
    args = parser.parse_args()

    # Load token from config
    HUGGINGFACE_TOKEN = load_config()

    if not os.path.exists(args.input_file):
        print(f"❌ Error: File '{args.input_file}' not found.")
        sys.exit(1)

    # Convert video files to audio
    input_file = convert_to_wav(args.input_file) if is_video_file(args.input_file) else args.input_file

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.use_whisperx:
        # Original Whisper + Pyannote pipeline
        print("🎙️ Running speaker diarization with Pyannote...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=HUGGINGFACE_TOKEN
        )
        pipeline = pipeline.to(device)
        diarization = pipeline(input_file, min_speakers=2, max_speakers=3)
        
        result, diarization = transcribe_with_whisper_pyannote(input_file, args.model_size, diarization)
        
        # Write results
        output_file = os.path.splitext(input_file)[0] + "_transcript.txt"
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
    else:
        # WhisperX pipeline
        result = transcribe_with_whisperx(input_file, args.model_size, HUGGINGFACE_TOKEN, device)
        
        # Write results
        output_file = os.path.splitext(input_file)[0] + "_transcript.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                start_time = segment['start']
                end_time = segment['end']
                speaker = segment.get('speaker', 'UNKNOWN')
                timestamp = f"[{start_time:.1f}s - {end_time:.1f}s]"
                text = segment['text'].strip()
                line = f"{timestamp} {speaker}: {text}"
                print(line)
                f.write(line + "\n")

    print(f"✅ Transcription with speaker diarization complete! Saved to {output_file}")

if __name__ == "__main__":
    main()