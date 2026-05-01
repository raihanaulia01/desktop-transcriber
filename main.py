from faster_whisper import WhisperModel
import soundcard as sc
import keyboard
import threading
import queue
import os

SAMPLE_RATE = 24000
CHUNK_SECONDS = 5

default_speaker = sc.default_speaker()
print(f"Using {default_speaker}")
mic = sc.get_microphone(default_speaker.id, include_loopback=True)

model_size = "medium"
print(f"Preparing model {model_size}...")
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

audio_queue = queue.Queue()

def transcribe_worker():
    while True:
        audio_mono = audio_queue.get()
        if audio_mono is None:
            break
        segments, info = model.transcribe(audio_mono, beam_size=5, task="translate")
        for segment in segments:
            print("    [%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

threading.Thread(target=transcribe_worker, daemon=True).start()

print("\nRecording... (press Q to stop)")
with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
    while not keyboard.is_pressed("q"):
        audio = recorder.record(numframes=SAMPLE_RATE * CHUNK_SECONDS)
        audio_queue.put(audio[:, 0])

audio_queue.put(None)  # signal transcribe worker to stop
print("Recording stopped.")
