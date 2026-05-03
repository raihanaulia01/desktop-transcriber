from faster_whisper import WhisperModel
import soundcard as sc
from rich.console import Console
from rich.table import Table
from rich.live import Live
from soundcard import SoundcardRuntimeWarning
import msvcrt
import argparse
import os
import threading
import queue
import numpy as np
from datetime import datetime, timedelta
import warnings
from constants import VALID_LANGUAGE_CODES, VALID_MODELS, VALID_DEVICES

# TODO soundcard library is volume-dependent. This makes the rms silence detection unreliable
#   ? change to logarithmic? rolling average? normalize volume (can't do this for every 0.1s chunks)? use silero-vad?
# TODO save rms debug values periodically

# sc spits out a warning when the script first starts. This is probably a windows issue. 
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

console = Console(highlight=False)

file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

script_dir = os.path.dirname(os.path.abspath(__file__))
rms_values_path = os.path.join(script_dir, f"rms_values_{file_timestamp}.csv")
transcribed_text_path = os.path.join(script_dir, f"transcribed_{file_timestamp}.txt")

parser = argparse.ArgumentParser(description="Live transcriber using faster_whisper by SYSTRAN")
parser.add_argument("--export-rms-values", action="store_true", help="Exports the rms values to a file in the script's directory.")
parser.add_argument("--export-transcribed", action="store_true", help="Exports the transcribed text to a file in the scrip's directory.")
parser.add_argument("--silence-duration", type=float, default=2.0, help="Seconds of silence before processing audio (default: 2.0).")
parser.add_argument("--silence-threshold", type=float, default=0.01, help="RMS threshold to consider a chunk silent (default: 0.01).")
parser.add_argument("--max-buffer-duration", type=int, default=20, help="Max seconds of audio to buffer before forcing processing (default: 20).")
parser.add_argument("--model", type=str, default="medium", choices=VALID_MODELS, help="Set the faster_whisper model, defaults to medium. Please check the vram requirements for each model before using.")
parser.add_argument("--device", type=str, default="cuda", choices=VALID_DEVICES, help="Set the device to use for computation, defaults to cuda.")
parser.add_argument("--language", type=str, default=None, choices=VALID_LANGUAGE_CODES, help="Language code (e.g. 'en', 'fr', 'ja'). If omitted, language will be auto-detected.")
arguments = parser.parse_args()

if arguments.export_rms_values:
    console.print(f"rms values will be exported to [magenta]{rms_values_path}[/magenta]")

transcribed_file = None
if arguments.export_transcribed:
    console.print(f"Transcribed text will be exported to [magenta]{transcribed_text_path}[/magenta]")
    transcribed_file = open(transcribed_text_path, "a")

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

def print_aligned_transcribe(timestamp, segment_start, segment_end, text):
    start = (timestamp + timedelta(seconds=segment_start)).strftime("%H:%M:%S.%f")
    end   = (timestamp + timedelta(seconds=segment_end)).strftime("%H:%M:%S.%f")
    timestamp_text = f"[{start[:-4]} -> {end[:-4]}]"
    table = Table(show_header=False, box=None)
    table.add_column(style="cyan", no_wrap=True, width=len(timestamp_text))
    table.add_column(style="white", ratio=1)

    table.add_row(timestamp_text, text)

    console.print(table)
    if transcribed_file:
        with open(transcribed_text_path, "a") as f:
            transcribed_file.write(f"{start[:-3]} -> {end[:-3]} : {text}\n")
            transcribed_file.flush()

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
            print_aligned_transcribe(timestamp, segment.start, segment.end, segment.text.strip())

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
with Live(console=console, refresh_per_second=10) as live:
    while True:
        # check 'q' keypress to exit
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key.lower() == "q":
                break

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
        live.update(f"{audio_time.strftime("%H:%M:%S.%f")[:-4]} | RMS = {rms:.3f} | Buffer = {len(audio_buffer):03d} | In queue: {audio_queue.qsize()}")

console.print("Recording stopped. Waiting for transcriber and recorder threads...")
stop_event.set()
recorder.join()
audio_queue.put(None)
transcriber.join()

if debug_rms_values:
    console.print(f"Exporting rms values to [magenta]{rms_values_path}[/magenta]")
    start_time = debug_rms_values[0][0]
    
    with open(rms_values_path, "a") as f:
        f.write("relative_time,absolute_time,rms_value\n")
        for value in debug_rms_values:
            relative_time = (value[0] - start_time).total_seconds() * 1000
            absolute_time = value[0].strftime("%H:%M:%S.%f")[:-3]
            f.write(f"{relative_time:.0f},{absolute_time},{value[1]}\n")

console.print("Done.")
os._exit(0)
