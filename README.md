# Audio Transcriber

A Windows-based live audio transcription project powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). It transcribes audio from the default speaker output.

## Features

- Live transcription of Windows speaker output through loopback capture
- Voice activity detection with [Silero VAD](https://github.com/snakers4/silero-vad)
- Automatic language detection or an explicitly selected language
- CUDA, CPU, and automatic device selection
- Configurable Whisper model, VAD threshold, silence duration, speech padding, and buffer size
- Optional timestamped text export for live transcription
- Filtering for several common Whisper hallucination phrases

## Requirements

- Windows
- Python 3.10 or newer
- A working Windows audio output device
- An NVIDIA GPU and compatible CUDA/PyTorch installation for GPU acceleration

CPU processing is supported but may be significantly slower, especially with larger models.

## Installation

Create and activate a virtual environment from the project directory:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the Python dependencies:

```powershell
pip install faster-whisper torch soundcard rich numpy
```

For CUDA support, install the PyTorch build that matches your CUDA installation. See the official PyTorch instructions at <https://pytorch.org/get-started/locally/>.

Whisper and Silero VAD model files are downloaded or loaded into these project directories on first use:

```text
whisper_model/
silero_model/
```

## Live Transcription

Start the live transcriber with the default settings:

```powershell
python main.py
```

The application captures audio from the default Windows speaker using loopback mode. It detects speech, sends completed speech segments to Whisper, and displays timestamped results in the terminal.

During recording:

- Press `Q` to stop recording.
- Press `Space` to force the current speech buffer to be transcribed.

The live transcriber uses Whisper's `translate` task, so recognized speech is translated to English. The `turbo` and `large-v3-turbo` models do not support translation and display a warning when selected.

### Live Transcription Options

| Option | Default | Description |
|---|---:|---|
| `--model` | `medium` | Whisper model to use |
| `--device` | `cuda` | Processing device: `cuda`, `cpu`, or `auto` |
| `--language` | automatic | Optional language code such as `en`, `fr`, or `ja` |
| `--silence-duration` | `0.5` | Seconds of silence before a segment is processed |
| `--vad-threshold` | `0.45` | Silero VAD speech probability threshold from `0.0` to `1.0` |
| `--speech-pad` | `50` | Speech padding in milliseconds |
| `--max-buffer-duration` | `20` | Maximum speech buffer duration in seconds |
| `--export-transcribed` | disabled | Write live transcription to a timestamped text file |
| `--export-rms-values` | disabled | Reserved option for RMS-value export |

Examples:

```powershell
python main.py --model small --device cuda
python main.py --device cpu --language en
python main.py --vad-threshold 0.55 --silence-duration 0.8
python main.py --export-transcribed
```

The supported model and language choices are defined in `constants.py`.

## Generated Files

Live transcription exports use timestamped names in the project directory:

```text
transcribed_YYYYMMDD_HHMMSS.txt
```

Whisper and Silero model caches are stored locally in `whisper_model/` and `silero_model/`.

## Audio Capture Notes

The live application uses the default speaker selected by Windows and calls `soundcard` with loopback enabled. It therefore transcribes audio playing through the selected output device, not microphone input by default.

To use a microphone instead, update the audio-device selection in `main.py`.

Changing or disconnecting the audio device while recording will stop the recorder thread. Restart the application after selecting a different Windows output device.
