# Pyannote Audio Transcription

A Python script for transcribing audio and video files with speaker diarization using Pyannote Audio and Whisper.

## Features
- Converts video files to audio automatically
- Performs speaker diarization using Pyannote Audio
- Transcribes audio using Whisper
- Outputs timestamped transcriptions with speaker labels

## Setup
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Hugging Face token:
```bash
HUGGINGFACE_TOKEN=<your_token_here>
```

## Usage
```bash
python transcribe.py <input_file> [--model-size <size>] [--use-whisperx]

Arguments:
  input_file            Audio or video file to transcribe
  --model-size          Model size (tiny, base, small, medium, large). Default: base
  --use-whisperx        Use WhisperX instead of Whisper+Pyannote for potentially better results
```

## Output
The script will output a timestamped transcript with speaker labels for each audio file in the `output` directory.

## Notes
- The script will automatically convert video files to audio using `ffmpeg`
- The script will use the `pyannote.audio` model for speaker diarization
- The script will use the `openai/whisper-large-v3` model for transcription
- The script will output a timestamped transcript with speaker labels for each audio file in the `output` directory

## TODO
- Add a way to specify the model size
- Add a way to specify the output directory
- Add a way to specify the input file
- Add a way to specify the output file
- Add a way to specify the output format
- Add a way to specify the output file     
