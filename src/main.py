import whisper
import pyaudio
import numpy as np

def capture_audio(model, chunk_size=1024, rate=16000, duration=5):
    """
    Capture audio from the microphone and transcribe it in real-time.
    Args:
        model: Whisper model to process audio.
        chunk_size: Number of audio frames per buffer.
        rate: Sampling rate of the microphone.
        duration: Duration to run the transcription (in seconds).
    """
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open microphone stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("Listening... (Press Ctrl+C to stop)")
    try:
        for _ in range(0, int(rate / chunk_size * duration)):
            # Read audio chunk
            audio_data = stream.read(chunk_size, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe chunk using Whisper
            result = model.transcribe(audio_array, fp16=False)
            print("Transcription:", result["text"])

    except KeyboardInterrupt:
        print("\nStopped listening.")
    finally:
        # Close stream
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    # Load Whisper model
    model_size = "base"  # Change to "tiny", "small", etc., if needed
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)

    # Start capturing and transcribing audio
    try:
        capture_audio(model, duration=30)  # Listen for 30 seconds
    except Exception as e:
        print(f"Error: {e}")
