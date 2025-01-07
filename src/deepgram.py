import asyncio
import websockets
import json
import wave
import pyaudio


def list_input_devices():
    p = pyaudio.PyAudio()
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            print(f"Index {i}: {device_info['name']}")
    p.terminate()


async def send_audio_to_deepgram(api_key, input_device_index, rate=16000, chunk_size=2048):
    url = "wss://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {api_key}"
    }

    async with websockets.connect(url, extra_headers=headers) as ws:
        print("Connected to Deepgram WebSocket")

        async def send_audio():
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=chunk_size
            )
            try:
                while True:
                    audio_data = stream.read(chunk_size, exception_on_overflow=False)
                    await ws.send(audio_data)
            except Exception as e:
                print(f"Audio streaming stopped: {e}")
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()

        async def receive_transcription():
            with open("transcription.txt", "a", encoding="utf-8") as f:
                while True:
                    try:
                        response = await ws.recv()
                        data = json.loads(response)
                        if "channel" in data and "alternatives" in data["channel"]:
                            transcription = data["channel"]["alternatives"][0]["transcript"]
                            print("Transcription:", transcription)
                            f.write(transcription + "\n")
                    except websockets.ConnectionClosed:
                        print("Connection to Deepgram closed.")
                        break
                    except Exception as e:
                        print(f"Error receiving transcription: {e}")

        await asyncio.gather(send_audio(), receive_transcription())


if __name__ == "__main__":
    # Step 1: List input devices
    list_input_devices()

    # Step 2: Set the microphone index
    device_index = int(input("Enter the device index for your microphone: "))

    # Step 3: Set your Deepgram API key
    api_key = input("Enter your Deepgram API key: ")

    # Step 4: Start capturing audio and sending it to Deepgram
    try:
        asyncio.run(send_audio_to_deepgram(api_key, input_device_index=device_index))
    except KeyboardInterrupt:
        print("\nStopped listening.")
    except Exception as e:
        print(f"Error: {e}")
