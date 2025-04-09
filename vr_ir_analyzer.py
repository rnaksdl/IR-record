from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
from pynput import keyboard
from datetime import datetime
import time
import os

# Initialize Picamera2
picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

# Set up encoder with high bitrate for good quality
encoder = None
output = None
recording = False
start_time = None
temp_filename = None

# Function to stop recording and rename the file
def stop_recording():
    global recording, start_time, temp_filename
    if recording:
        picam2.stop_encoder()
        recording = False
        end_time = time.time()
        duration = int(end_time - start_time)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_filename = f"video_{timestamp}_{duration}s.mp4"
        os.rename(temp_filename, final_filename)
        print(f"Recording stopped. Saved as {final_filename}")

def on_press(key):
    global recording, start_time, temp_filename, encoder, output
    try:
        if key.char == 'r' and not recording:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            temp_filename = f"temp_{timestamp}.mp4"
            encoder = H264Encoder(bitrate=10000000)
            output = FfmpegOutput(temp_filename)
            picam2.start_encoder(encoder, output)
            recording = True
            print(f"Recording started at {timestamp}...")

        elif key.char == 's' and recording:
            stop_recording()

        elif key.char == 'q':
            print("Exiting...")
            stop_recording()
            return False  # Stop listener

    except AttributeError:
        pass  # Ignore special keys

print("Press 'r' to start recording, 's' to stop recording, and 'q' to quit.")

picam2.start()

# Start the keyboard listener
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

picam2.stop()
