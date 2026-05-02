from faster_whisper import WhisperModel
import soundcard as sc
from rich.console import Console
from rich.table import Table
from soundcard import SoundcardRuntimeWarning
import keyboard
import argparse
import os
import threading
import queue
import numpy as np
from datetime import datetime, timedelta
import warnings
from constants import VALID_LANGUAGE_CODES, VALID_MODELS, VALID_DEVICES

# sc spits out a warning when the script first starts. This is probably a windows issue. 
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

console = Console(highlight=False)

def print_aligned_transcribe(timestamp, text):
    table = Table(show_header=False, box=None)
    table.add_column(style="cyan", no_wrap=True, width=len(timestamp))
    table.add_column(style="white", ratio=1)

    table.add_row(timestamp, text)

    console.print(table)

script_dir = os.path.dirname(os.path.abspath(__file__))
default_path = os.path.join(script_dir, f"rms_values_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv")

parser = argparse.ArgumentParser(description="Live transcriber using faster_whisper by SYSTRAN")
parser.add_argument("--export-rms-values", action="store_true", help="Exports the rms values to a file in the script's directory.")
parser.add_argument("--silence-duration", type=float, default=2.0, help="Seconds of silence before processing audio (default: 2.0)")
parser.add_argument("--silence-threshold", type=float, default=0.01, help="RMS threshold to consider a chunk silent (default: 0.01)")
parser.add_argument("--max-buffer-duration", type=int, default=20, help="Max seconds of audio to buffer before forcing processing (default: 20)")
parser.add_argument("--model", type=str, default="medium", choices=VALID_MODELS, help="Set the faster_whisper model, defaults to medium. Please check the vram requirements for each model before using.")
parser.add_argument("--device", type=str, default="cuda", choices=VALID_DEVICES, help="Set the device to use for computation, defaults to cuda")
parser.add_argument("--language", type=str, default=None, choices=VALID_LANGUAGE_CODES, help="Language code (e.g. 'en', 'fr', 'ja'). If omitted, language will be auto-detected.")
arguments = parser.parse_args()

if arguments.export_rms_values:
    console.print(f"rms values will be exported to [magenta]{default_path}[/magenta]")

if arguments.model == "turbo" or arguments.model ==  "large-v3-turbo":
    console.print(f"Warning: the turbo model doesn't support translation.")

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = arguments.silence_threshold
SILENCE_DURATION = arguments.silence_duration
CHUNK_SECONDS = 0.1
MAX_BUFFER_SECONDS = arguments.max_buffer_duration
SILENCE_CHUNKS_NEEDED = int(SILENCE_DURATION / CHUNK_SECONDS)

default_speaker = sc.default_speaker()
console.print(f"Using [magenta]{default_speaker}[/magenta]")
mic = sc.get_microphone(default_speaker.id, include_loopback=True)

model_size = arguments.model
console.print(f"Preparing model {model_size}...")
model_directory = os.path.join(script_dir, "whisper_model/")
model = WhisperModel(model_size, device=arguments.device, compute_type="int8_float16", download_root=model_directory)
console.print(f"Model {model_size} ready!\n", style="green")

audio_queue = queue.Queue()

def transcribe_worker():
    while True:
        data = audio_queue.get()
        if data is None:
            break
        audio_mono, timestamp = data
        segments, _ = model.transcribe(
            audio_mono, beam_size=5, 
            task="translate", 
            vad_filter=True, 
            language=arguments.language
        )
        for segment in segments:
            start = (timestamp + timedelta(seconds=segment.start)).strftime("%H:%M:%S.%f")[:-4]
            end   = (timestamp + timedelta(seconds=segment.end)).strftime("%H:%M:%S.%f")[:-4]
            print_aligned_transcribe(f"[{start} -> {end}]", segment.text.strip())

raw_audio_queue = queue.Queue()
def recorder_thread():
    with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
        while not stop_event.is_set():
            audio = recorder.record(numframes=SAMPLE_RATE * CHUNK_SECONDS)
            raw_audio_queue.put(audio)

stop_event = threading.Event()
recorder = threading.Thread(target=recorder_thread, daemon=True)
recorder.start()

transcriber = threading.Thread(target=transcribe_worker, daemon=True)
transcriber.start()

silence_frames = 0
audio_buffer = []
debug_rms_values = [] if arguments.export_rms_values else None

console.print("Recording... (press Q to stop)")
while not keyboard.is_pressed("q"):
    audio = raw_audio_queue.get()
    audio_mono = audio[:, 0]
    audio_time = datetime.now()
    audio_buffer.append((audio_mono, audio_time))
    rms = np.sqrt(np.mean(audio_mono**2))
    is_silent = rms < SILENCE_THRESHOLD
    
    if not debug_rms_values is None:
        debug_rms_values.append((audio_time, rms))

    if is_silent:
        silence_frames += 1
    else:
        silence_frames = 0

    if (silence_frames >= SILENCE_CHUNKS_NEEDED and len(audio_buffer) > 0) or (
            len(audio_buffer) > int(MAX_BUFFER_SECONDS/CHUNK_SECONDS)):
        full_audio = np.concatenate([audios[0] for audios in audio_buffer])
        audio_queue.put((full_audio, audio_buffer[0][1]))
        audio_buffer = []
        silence_frames = 0

console.print("Recording stopped. Waiting for transcriber and recorder threads...")
stop_event.set()
recorder.join()
audio_queue.put(None)
transcriber.join()

if debug_rms_values:
    console.print(f"Exporting rms values to [magenta]{default_path}[/magenta]")
    start_time = debug_rms_values[0][0]
    
    with open(default_path, "a") as f:
        f.write("relative_time,absolute_time,rms_value\n")
        for value in debug_rms_values:
            relative_time = (value[0] - start_time).total_seconds() * 1000
            absolute_time = value[0].strftime("%H:%M:%S.%f")[:-3]
            f.write(f"{relative_time:.0f},{absolute_time},{value[1]}\n")

console.print("Done.")
os._exit(0)
