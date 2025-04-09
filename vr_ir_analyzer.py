from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
from gpiozero import Button
from datetime import datetime
import time
import os

# Initialize Picamera2
picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

encoder = None
output = None
recording = False
start_time = None
temp_filename = None

# Define GPIO pins for buttons
record_button = Button(17)  # GPIO17 to start/stop recording
quit_button = Button(27)    # GPIO27 to quit script

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

print("Press the record button to start/stop recording, and quit button to exit.")

try:
    picam2.start()
    while True:
        if record_button.is_pressed and not recording:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            temp_filename = f"temp_{timestamp}.mp4"
            encoder = H264Encoder(bitrate=10000000)
            output = FfmpegOutput(temp_filename)
            picam2.start_encoder(encoder, output)
            recording = True
            print(f"Recording started at {timestamp}...")
            time.sleep(0.5)  # Debounce delay

        elif record_button.is_pressed and recording:
            stop_recording()
            time.sleep(0.5)  # Debounce delay

        elif quit_button.is_pressed:
            print("Exiting...")
            stop_recording()
            break

        time.sleep(0.1)

finally:
    picam2.stop()
