import RPi.GPIO as GPIO
import time
import threading
import queue
import sounddevice as sd
import soundfile as sf
import subprocess
import re
from datetime import datetime, timedelta
from llama_cpp import Llama
from pydub import AudioSegment
import numpy as np
from vosk import Model, KaldiRecognizer
from scipy.signal import resample
import os

os.environ["LLAMA_CPP_FORCE_DISABLE_GPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_DISABLE_GPU"] = "1"
os.environ["ONNX_RUNTIME_DISABLE_GPU"] = "1"

# ----- CONFIGURATION -----
MOUTH_PIN = 18
LIGHT_PIN = 23
WAKE_WORDS = ["hey jack o'lantern", "hey pumpkin"]
SILENCE_TIMEOUT = 180  # seconds
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ""
env["ORT_DISABLE_GPU"] = "1"
env["ONNX_RUNTIME_DISABLE_GPU"] = "1"


# Piper configuration
PIPER_COMMAND = [
    "piper",
    "--model", os.path.join(PARENT_DIR,"voices","en_GB-semaine-medium.onnx"),  # adjust if needed
    "--output-raw",
    "--use-cpu",
    "--speaker", "1"
]



# Llama configuration
LLAMA_MODEL_PATH = os.path.join(PARENT_DIR,"llama.cpp","models","LiquidAI_LFM2-2.6B-GGUF_LFM2-2.6B-Q4_K_M.gguf")
PROMPT_INSTRUCTIONS = """You a ghost who has been doomed to haunt this pumpkin for eternity.
Respond in a spooky, forlorn way. Keep responses short - two sentences max. Do not use bad words. 
NEVER describe your actions or include anything in *asterisks*. Only speak out loud."""

# --------------------------

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(MOUTH_PIN, GPIO.OUT)
GPIO.setup(LIGHT_PIN, GPIO.OUT)
GPIO.output(MOUTH_PIN, GPIO.LOW)
GPIO.output(LIGHT_PIN, GPIO.LOW)

# Samplerate settings
MIC_RATE = 44100
VOSK_RATE = 16000

# Initialize AI and ASR
vosk_model = Model(os.path.join(PARENT_DIR,"ttsmodels"))
llama = Llama(
    model_path=LLAMA_MODEL_PATH,
    chat_format="chatml",  # or "llama-2" "chatml", "openchat", depending on the model,
    n_gpu_layers=0
)

llama.reset()  # Only at startup

conversation = [
    {"role": "system", "content": PROMPT_INSTRUCTIONS},
]

last_active_time = datetime.now()
listening = True


def speak(text):
    # Ignore actions in asterisks
    clean_text = re.sub(r"\*", " ", text)
    
    # Light on
    GPIO.output(LIGHT_PIN, GPIO.HIGH)

    # Generate speech audio with Piper
    p = subprocess.Popen(
        PIPER_COMMAND,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env
    )
    p.stdin.write(clean_text.encode())
    p.stdin.close()

    # Read all raw PCM bytes from stdout
    raw_bytes = p.stdout.read()
    p.stdout.close()
    p.wait()

    samplerate = 22050
    channels = 1
    dtype = np.int16
    
    audio_data = np.frombuffer(raw_bytes, dtype=dtype)
    p.stdout.close()

    # Estimate syllables (basic method)
    syllables = len(re.findall(r'[aeiouy]+', clean_text.lower()))
    duration = len(audio_data) / samplerate
    mouth_thread = threading.Thread(target=move_mouth, args=(syllables, duration))
    mouth_thread.start()

    # Play audio
    sd.play(audio_data, samplerate)
    sd.wait()

    mouth_thread.join()

    # Light off
    GPIO.output(LIGHT_PIN, GPIO.LOW)


def move_mouth(syllables, duration):
    if syllables == 0:
        return
    interval = duration / syllables
    for _ in range(syllables):
        GPIO.output(MOUTH_PIN, GPIO.HIGH)
        time.sleep(interval / 2)
        GPIO.output(MOUTH_PIN, GPIO.LOW)
        time.sleep(interval / 2)


def listen_for_command():
    global last_active_time

    duration = 7
    channels = 1

    print("Listening...")
    recording = sd.rec(int(duration * MIC_RATE), samplerate=MIC_RATE, channels=channels, dtype='int16')
    sd.wait() 

    if MIC_RATE != VOSK_RATE:
        recording_float = recording.astype(np.float32)
        num_samples = int(len(recording_float) * VOSK_RATE / MIC_RATE)
        resampled = resample(recording_float, num_samples)
        audio_data = resampled.astype(np.int16).tobytes()
    else:
        audio_data = recording.tobytes()

    recognizer = KaldiRecognizer(vosk_model, VOSK_RATE)

    if recognizer.AcceptWaveform(audio_data):
        result = recognizer.Result()
        text = re.search(r'"text" : "([^"]*)"', result)
        if text:
            recognized = text.group(1).strip().lower()
            print("Heard:", recognized)
            return recognized
    else:
        partial = recognizer.PartialResult()
        print("Partial:", partial)
        return partial
    return "" 



def generate_response(user_text):
    conversation.append({"role": "user", "content": user_text})
    output_text = []
    speak_buffer = []

    response_stream = llama.create_chat_completion(
        messages=conversation,
        max_tokens=60,
        temperature=0.7,
        stream=True
    )

    print()  # Newline for readability

    for chunk in response_stream:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            token = delta["content"]
            print(token, end='', flush=True)

            output_text.append(token)
            speak_buffer.append(token)

            # Check for sentence-ending punctuation
            if token in [".", ",", "!"]:
                sentence = "".join(speak_buffer).strip()
                if sentence:
                    speak(sentence)
                speak_buffer = []  # Clear buffer after speaking

    # Speak any remaining text if punctuation never appeared
    if speak_buffer:
        sentence = "".join(speak_buffer).strip()
        if sentence:
            speak(sentence)

    message = "".join(output_text).strip()
    print()  # newline after response

    conversation.append({"role": "assistant", "content": message})
    return message


def wake_word_detected(text):
    text = text.lower()
    return any(word in text for word in WAKE_WORDS)


def reset_inactivity_timer():
    global last_active_time
    last_active_time = datetime.now()


def inactivity_monitor():
    global listening
    while True:
        if datetime.now() - last_active_time > timedelta(seconds=SILENCE_TIMEOUT):
            print("Going to sleep...")
            listening = False
        time.sleep(10)


def main():
    global listening
    threading.Thread(target=inactivity_monitor, daemon=True).start()
    print("Jack O'Lantern is ready!")

    while True:
        if not listening:
            text = listen_for_command()
            if wake_word_detected(text):
                print("Wake word detected.")
                reset_inactivity_timer()
                listening = True
            continue

        text = listen_for_command()
        if wake_word_detected(text):
            print("Already awake.")
            reset_inactivity_timer()
            continue

        if text:
            reset_inactivity_timer()
            response = generate_response(text)


try:
    main()
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    GPIO.cleanup()
