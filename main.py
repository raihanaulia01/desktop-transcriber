from faster_whisper import WhisperModel
import os

model_size = "medium"
print(f"Preparing model {model_size}...")
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

def test_sample():
    sample_files = os.listdir("sample_audio")
    print(f"Sample audio files: {sample_files}")
    for audio_file in sample_files:
        audio_path = f"sample_audio/{audio_file}"
        print(f"Transcribing {audio_path}")
        segments, info = model.transcribe(audio_path, beam_size=5, task="translate")
        print("    Detected language '%s' with probability %f" % (info.language, info.language_probability))

        for segment in segments:
            print("    [%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        print()

test_sample()