from faster_whisper import WhisperModel
import soundcard as sc
import keyboard
import threading
import queue
import numpy as np
from datetime import datetime, timedelta
# import os

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
        audio_mono, timestamp = audio_queue.get()
        if audio_mono is None:
            break
        segments, info = model.transcribe(audio_mono, beam_size=5, task="translate", vad_filter=True)
        for segment in segments:
            start = (timestamp + timedelta(seconds=segment.start)).strftime("%H:%M:%S.%f")[:-2]
            end   = (timestamp + timedelta(seconds=segment.end)).strftime("%H:%M:%S.%f")[:-2]
            print(f"    [{start} -> {end}] {segment.text}")

threading.Thread(target=transcribe_worker, daemon=True).start()

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 2
CHUNK_SECONDS = 0.1

silence_frames = 0
audio_buffer = []
# debug_rms_values = []
MAX_BUFFER_SECONDS = 20
SILENCE_CHUNKS_NEEDED = int(SILENCE_DURATION / CHUNK_SECONDS)

print("\nRecording... (press Q to stop)")
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


audio_queue.put(None)  # signal transcribe worker to stop
print("Recording stopped.")

# with open("rmsvalues.txt", "a") as f:
#     for value in debug_rms_values:
#         f.write(f"{value}\n")
