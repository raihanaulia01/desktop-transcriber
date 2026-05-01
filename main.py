from faster_whisper import WhisperModel
import soundcard as sc
from soundcard import SoundcardRuntimeWarning
import keyboard
import threading
import queue
import numpy as np
from datetime import datetime, timedelta
import warnings

# sc spits out a warning when the script first starts. This is probably a windows issue. 
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

default_speaker = sc.default_speaker()
print(f"Using {default_speaker}")
mic = sc.get_microphone(default_speaker.id, include_loopback=True)

model_size = "medium"
print(f"Preparing model {model_size}...")
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
print(f"Model {model_size} ready!\n")

audio_queue = queue.Queue()

def transcribe_worker():
    while True:
        data = audio_queue.get()
        if data is None:
            break
        audio_mono, timestamp = data
        segments, info = model.transcribe(audio_mono, beam_size=5, task="translate", vad_filter=True)
        for segment in segments:
            start = (timestamp + timedelta(seconds=segment.start)).strftime("%H:%M:%S.%f")[:-4]
            end   = (timestamp + timedelta(seconds=segment.end)).strftime("%H:%M:%S.%f")[:-4]
            print(f"  [{start} -> {end}] {segment.text}")

transcriber = threading.Thread(target=transcribe_worker, daemon=True)
transcriber.start()

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 2
CHUNK_SECONDS = 0.1

silence_frames = 0
audio_buffer = []
# debug_rms_values = []
MAX_BUFFER_SECONDS = 20
SILENCE_CHUNKS_NEEDED = int(SILENCE_DURATION / CHUNK_SECONDS)

print("Recording... (press Q to stop)")
with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
    while not keyboard.is_pressed("q"):
        audio = recorder.record(numframes=SAMPLE_RATE * CHUNK_SECONDS)
        audio_mono = audio[:, 0]
        audio_buffer.append((audio_mono, datetime.now()))
        rms = np.sqrt(np.mean(audio_mono**2))
        # print(rms)
        # debug_rms_values.append(rms)
        is_silent = rms < SILENCE_THRESHOLD

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


print("Recording stopped. Waiting for transcriber thread...")
audio_queue.put(None)
transcriber.join()
print("Done.")

# with open("rmsvalues.txt", "a") as f:
#     for value in debug_rms_values:
#         f.write(f"{value}\n")
