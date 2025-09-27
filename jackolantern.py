import os
import time
import subprocess
import RPi.GPIO as GPIO
import re
import threading

# === GPIO Setup ===
LIGHTS_GPIO = 23
MOUTH_GPIO = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(LIGHTS_GPIO, GPIO.OUT)
GPIO.setup(MOUTH_GPIO, GPIO.OUT)
GPIO.output(LIGHTS_GPIO, GPIO.LOW)
GPIO.output(MOUTH_GPIO, GPIO.LOW)

# === Settings ===
WAKE_WORD = "hey jack o lantern"
IDLE_TIMEOUT = 300  # 5 minutes

# === Piper TTS Settings ===
PIPER_MODEL_PATH = "/home/pi/models/en_US-l2arctic-medium.onnx"
PIPER_SPEAKER = "SVBI"

# === Paths to AI tools ===
WHISPER_MODEL = "models/ggml-base.en.bin"
LLAMA_MODEL = "/home/pi/models/your-model.gguf"

# === Record audio input to .wav ===
def record_audio(filename="input.wav", duration=5):
    subprocess.run(["arecord", "-D", "plughw:1", "-f", "cd", "-t", "wav", "-d", str(duration), filename], check=True)

# === Transcribe using Whisper ===
def transcribe_audio(file_path):
    result = subprocess.run(
        ["./main", "-m", WHISPER_MODEL, "-f", file_path],
        capture_output=True,
        text=True,
        cwd="/home/pi/whisper.cpp"
    )
    return result.stdout.strip().lower()

# === Generate LLaMA response ===
def get_llama_response(prompt):
    result = subprocess.run(
        ["./main", "-m", LLAMA_MODEL, "-p", prompt, "-n", "200", "-t", "4"],
        capture_output=True,
        text=True,
        cwd="/home/pi/llama.cpp/build"
    )
    return result.stdout.strip()

# === Mouth animation per syllable (run in parallel) ===
def animate_mouth(text):
    syllables = re.findall(r'[aeiouy]+', text, re.I)
    duration = 0.25  # time per syllable, tweak as needed
    for _ in syllables:
        GPIO.output(MOUTH_GPIO, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(MOUTH_GPIO, GPIO.LOW)
        time.sleep(duration)

# === Speak using Piper in real time (no .wav files) ===
def speak_with_mouth(text):
    print("🔊 Speaking...")

    # Turn on lights
    GPIO.output(LIGHTS_GPIO, GPIO.HIGH)

    # Start mouth animation thread
    mouth_thread = threading.Thread(target=animate_mouth, args=(text,))
    mouth_thread.start()

    # Start Piper and stream audio directly to aplay
    piper_proc = subprocess.Popen(
        [
            "/usr/bin/piper",
            "--model", PIPER_MODEL_PATH,
            "--speaker", PIPER_SPEAKER,
            "--output_raw",
            "--text", text
        ],
        stdout=subprocess.PIPE
    )

    aplay_proc = subprocess.Popen(
        ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw"],
        stdin=piper_proc.stdout
    )

    aplay_proc.wait()
    mouth_thread.join()

    # Turn off lights
    GPIO.output(LIGHTS_GPIO, GPIO.LOW)
    print("✅ Done speaking.")

# === Main Loop ===
def main():
    print("🎃 Jack O'Lantern is sleeping...")

    while True:
        record_audio("wake.wav", duration=4)
        transcript = transcribe_audio("wake.wav")

        if WAKE_WORD in transcript:
            print("🎃 Wake word detected!")
            last_active = time.time()

            while True:
                print("🎤 Listening for user input...")
                record_audio("user.wav", duration=6)
                user_input = transcribe_audio("user.wav")

                if not user_input.strip():
                    if time.time() - last_active > IDLE_TIMEOUT:
                        print("💤 Going back to sleep...")
                        break
                    else:
                        continue

                print(f"🗣️ You said: {user_input}")
                last_active = time.time()

                reply = get_llama_response(user_input)
                print(f"🎃 Jack says: {reply}")

                speak_with_mouth(reply)

        else:
            print("💤 No wake word detected.")

# === Run It ===
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("🛑 Exiting...")
    finally:
        GPIO.cleanup()
