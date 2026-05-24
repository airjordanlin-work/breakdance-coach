from kokoro_onnx import Kokoro
import soundfile as sf
import subprocess
import tempfile

kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
samples, sample_rate = kokoro.create("Perfect, great form", voice="af_heart", speed=1.0)

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    sf.write(f.name, samples, sample_rate)
    subprocess.run(["afplay", f.name])

print("Done")