import pyaudio
import whisper
import numpy as np

def list_input_devices():
    p = pyaudio.PyAudio()
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            print(f"Index {i}: {device_info['name']}")
    p.terminate()

def capture_audio(model, chunk_size=1024, rate=16000, duration=5, input_device_index=None):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=chunk_size)

    print("Listening... (Press Ctrl+C to stop)")
    try:
        with open("transcription.txt", "a", encoding="utf-8") as f:  # Open file in UTF-8 encoding
            for _ in range(0, int(rate / chunk_size * duration)):
                audio_data = stream.read(chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Restrict transcription to English and save to file
                result = model.transcribe(audio_array, fp16=False, language="en")
                transcription = result["text"]
                print("Transcription:", transcription)
                f.write(transcription + "\n")  # Append to the file
    except KeyboardInterrupt:
        print("\nStopped listening.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    # Step 1: List input devices
    list_input_devices()
    
    # Step 2: Choose a model
    model_size = "base"
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)

    # Step 3: Set the microphone index
    device_index = int(input("Enter the device index for your microphone: "))

    # Step 4: Start capturing audio
    try:
        capture_audio(model, duration=10, input_device_index=device_index)
    except Exception as e:
        print(f"Error: {e}")
