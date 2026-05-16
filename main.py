from faster_whisper import WhisperModel
import torch
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

# sc spits out a warning when the script first starts. This is probably a windows issue. 
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

console = Console(highlight=False)

file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

script_dir = os.path.dirname(os.path.abspath(__file__))
rms_values_path = os.path.join(script_dir, f"rms_values_{file_timestamp}.csv")
transcribed_text_path = os.path.join(script_dir, f"transcribed_{file_timestamp}.txt")

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x!r} not a floating-point literal")

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x!r} not in range [0.0, 1.0]")
    return x


parser = argparse.ArgumentParser(description="Live transcriber using faster_whisper by SYSTRAN")
parser.add_argument("--export-rms-values", action="store_true", help="Exports the rms values to a file in the script's directory.")
parser.add_argument("--export-transcribed", action="store_true", help="Exports the transcribed text to a file in the scrip's directory.")
parser.add_argument("--silence-duration", type=float, default=0.5, help="Seconds of silence before processing audio (default: 0.5).")
parser.add_argument("--vad-threshold", type=restricted_float, default=0.5, help="Silero VAD speech probability sensitivity threshold between 0.0 and 1.0 (default: 0.5).")
parser.add_argument("--max-buffer-duration", type=int, default=20, help="Max seconds of audio to buffer before forcing processing (default: 20).")
parser.add_argument("--model", type=str, default="medium", choices=VALID_MODELS, help="Set the faster_whisper model, defaults to medium. Please check the vram requirements for each model before using.")
parser.add_argument("--device", type=str, default="cuda", choices=VALID_DEVICES, help="Set the device to use for computation, defaults to cuda.")
parser.add_argument("--language", type=str, default=None, choices=VALID_LANGUAGE_CODES, help="Language code (e.g. 'en', 'fr', 'ja'). If omitted, language will be auto-detected.")
arguments = parser.parse_args()

transcribed_file = None
if arguments.export_transcribed:
    console.print(f"Transcribed text will be exported to [magenta]{transcribed_text_path}[/magenta]")
    transcribed_file = open(transcribed_text_path, "a", encoding="utf-8")

if arguments.model == "turbo" or arguments.model ==  "large-v3-turbo":
    console.print(f"Warning: the turbo model doesn't support translation.")

SAMPLE_RATE = 16000
SILENCE_DURATION = arguments.silence_duration
CHUNK_SECONDS = 0.032
MAX_BUFFER_SECONDS = arguments.max_buffer_duration

# set speaker
default_speaker = sc.default_speaker()
console.print(f"Using [magenta]{default_speaker}[/magenta]")
mic = sc.get_microphone(default_speaker.id, include_loopback=True)

# initialize faster whisper model
model_size = arguments.model
console.print(f"Preparing model {model_size}...")
model_directory = os.path.join(script_dir, "whisper_model/")
os.makedirs(model_directory, exist_ok=True)
model = WhisperModel(model_size, device=arguments.device, compute_type="int8_float16", download_root=model_directory)
console.print(f"Model {model_size} ready!\n", style="green")

# initialize silero vad
console.print("Preparing Silero VAD...")
vad_model_directory =  os.path.join(script_dir, "silero_model/")
os.makedirs(vad_model_directory, exist_ok=True)
torch.hub.set_dir(vad_model_directory)

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True) # type: ignore
VADIterator = utils[3]
vad_iterator = VADIterator(
    vad_model, 
    threshold=arguments.vad_threshold, 
    sampling_rate=SAMPLE_RATE, 
    min_silence_duration_ms=int(SILENCE_DURATION * 1000)
)
console.print("Silero VAD engine live!\n", style="green")

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
        transcribed_file.write(f"{start[:-3]} -> {end[:-3]} : {text}\n")
        transcribed_file.flush()

audio_queue = queue.Queue()
def transcribe_worker():
    while not stop_event.is_set():
        data = audio_queue.get()
        if data is None:
            break
        audio_mono, timestamp = data
        segments, _ = model.transcribe(
            audio_mono, beam_size=5, 
            task="translate", 
            language=arguments.language
        )
        for segment in segments:
            print_aligned_transcribe(timestamp, segment.start, segment.end, segment.text.strip())

# TODO add a retry mechanism to automatically find and reconnect the audio device
raw_audio_queue = queue.Queue()
def recorder_thread():
    try:
        # ? maybe put the get microphone here, then add another loop below this as the actual recording loop
        # ? and move the stop event to the top, and move the mic recorder under the stop event loop
        with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
            while not stop_event.is_set():
                try:
                    audio = recorder.record(numframes=SAMPLE_RATE * CHUNK_SECONDS)
                    raw_audio_queue.put(audio)
                except RuntimeError as e:
                    if "0x88890004" in str(e) or "0x10000000" in str(e):
                        console.print("\n[bold red]Audio device disconnected or changed! Stopping recorder thread.[/bold red]")
                        stop_event.set()
                        break
                    else:
                        raise e
    except Exception as e:
        console.print(f"[bold red]Fatal error in recorder thread:[/bold red] {e}", highlight=True)
        stop_event.set()

stop_event = threading.Event()
recorder = threading.Thread(target=recorder_thread, daemon=True)
recorder.start()

transcriber = threading.Thread(target=transcribe_worker, daemon=True)
transcriber.start()

silence_frames = 0
audio_buffer = []
is_speaking = False

console.print("Recording...")
console.print("Press Q to stop, press SPACE to force transcribe")
with Live(console=console, refresh_per_second=10) as live:
    while not stop_event.is_set():
        # check 'q' keypress to exit
        key = None
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key.lower() == "q":
                break

        try:
            audio = raw_audio_queue.get(timeout=1)
        except queue.Empty:
            continue
        audio_mono = audio[:, 0]
        audio_time = datetime.now()
        audio_tensor = torch.from_numpy(audio_mono)
        vad_output = vad_iterator(audio_tensor)

        if vad_output:
            if "start" in vad_output:
                is_speaking = True
                audio_buffer = [(audio_mono, audio_time)]
            
            if "end" in vad_output:
                if is_speaking and audio_buffer:
                    full_audio = np.concatenate([audios[0] for audios in audio_buffer])
                    audio_queue.put((full_audio, audio_buffer[0][1]))
                is_speaking = False
                audio_buffer = []
                vad_iterator.reset_states()
        else:
            if is_speaking:
                audio_buffer.append((audio_mono, audio_time))

        reached_max_buffer = len(audio_buffer) > int(MAX_BUFFER_SECONDS/CHUNK_SECONDS)
        force_transcribe = key == ' '

        if reached_max_buffer or force_transcribe:
            if is_speaking and audio_buffer:
                full_audio = np.concatenate([audios[0] for audios in audio_buffer])
                audio_queue.put((full_audio, audio_buffer[0][1]))
            is_speaking = False
            audio_buffer = []
            vad_iterator.reset_states()
        status_text = "Speaking " if is_speaking else "Listening"
        live.update(f"{audio_time.strftime("%H:%M:%S.%f")[:-4]} | {status_text} | Buffer Seconds = {round(len(audio_buffer)*CHUNK_SECONDS, 1)} | In queue: {audio_queue.qsize()}")

console.print("Recording stopped. Waiting for transcriber and recorder threads...")
stop_event.set()
recorder.join()
audio_queue.put(None)
transcriber.join()

console.print("Done.")
os._exit(0)
