import PySimpleGUI as sg
import pyaudio
import wave
import os
import tempfile
import time
import threading
from util import predict


def record_audio(duration=3, sample_rate=44100, channels=1, chunk=1024):
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk,
    )

    print("Recording...")
    frames = []
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return frames, sample_rate

def save_wav(frames, sample_rate):
    temp_file = os.path.join("./", "temp_audio.wav")

    wf = wave.open(temp_file, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()

    return temp_file


def process_audio_with_progress(audio_file, window):
    progress_bar = window["-PROGRESS-"]
    for i in range(100):
        time.sleep(
            0.05
        )  # Adjust this value to match your function's actual processing time
        progress_bar.update(current_count=i + 1)
    result = predict(audio_file)
    window.write_event_value("-FUNCTION-DONE-", result)


def animate_text(window, key, text):
    for i in range(len(text) + 1):
        window[key].update(text[:i])
        time.sleep(0.05)


sg.theme_background_color("#2A2A72")  # Navy background
sg.theme_text_element_background_color("#2A2A72")  # Navy
sg.theme_input_background_color("#4B0082")  # Indigo for input fields
sg.theme_text_color("#E6E6FA")  # Lavender text
sg.theme_button_color(("#E6E6FA", "#4B0082"))  # Lavender text on Indigo button

layout = [
    [
        sg.Text(
            "Click the button to record audio",
            font=("Helvetica", 14),
            text_color="#E6E6FA",
        )
    ],
    [
        sg.Button(
            "Record",
            size=(10, 1),
            font=("Helvetica", 12),
            button_color=("#E6E6FA", "#4B0082"),
        )
    ],
    [
        sg.ProgressBar(
            100,
            orientation="h",
            size=(20, 20),
            key="-PROGRESS-",
            visible=False,
            bar_color=("#9370DB", "#2A2A72"),
        )
    ],
    [
        sg.Text(
            "",
            key="-OUTPUT-",
            size=(50, 3),
            font=("Helvetica", 12),
            text_color="#E6E6FA",
        )
    ],
]

window = sg.Window("AI Audio-Based Emotion Detector", layout, finalize=True)
