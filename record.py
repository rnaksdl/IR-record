#!/usr/bin/env python3

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from picamera2.previews.qt import QGlPicamera2
from libcamera import Transform
import time
import os
from datetime import datetime
import shutil
import threading
import sys

output_folder = "recordings"
os.makedirs(output_folder, exist_ok=True)

# Initialize camera with horizontal flip (mirror)
picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": 60.0},
    transform=Transform(hflip=1)
)
picam2.configure(video_config)

encoder = H264Encoder(bitrate=4000000)

recording = False
temp_filename = ""
start_time = 0

# Start camera and Qt-based preview (avoids event loop conflict)
picam2.start()
preview = QGlPicamera2(picam2)
preview.start()

print("IR Signal Analysis Recording System")
print("Commands:")
print("  1 - Start recording")
print("  2 - Stop recording")
print("  3 - Quit")

# Function to display recording duration
def show_duration():
    while recording:
        elapsed = time.time() - start_time
        print(f"\rRecording duration: {elapsed:.1f}s", end='', flush=True)
        time.sleep(0.5)

try:
    while True:
        command = input("\n> ").strip()

        if command == "1" and not recording:
            temp_filename = f"{output_folder}/temp_recording.h264"
            picam2.start_recording(encoder, FileOutput(temp_filename))
            recording = True
            start_time = time.time()
            print(f"Recording started at {datetime.now().strftime('%H:%M:%S')}")

            # Start duration display thread
            duration_thread = threading.Thread(target=show_duration, daemon=True)
            duration_thread.start()

        elif command == "2" and recording:
            picam2.stop_recording()
            recording = False
            actual_duration = time.time() - start_time
            final_filename = f"{output_folder}/{datetime.now().strftime('%y%m%d_%H%M%S')}_{round(actual_duration)}s.h264"
            shutil.move(temp_filename, final_filename)
            print(f"\nRecording stopped. Duration: {actual_duration:.1f}s. Saved as {final_filename}")

        elif command == "3":
            if recording:
                picam2.stop_recording()
                recording = False
                actual_duration = time.time() - start_time
                final_filename = f"{output_folder}/{datetime.now().strftime('%y%m%d_%H%M%S')}_{round(actual_duration)}s.h264"
                shutil.move(temp_filename, final_filename)
                print(f"\nRecording stopped. Duration: {actual_duration:.1f}s. Saved as {final_filename}")
            print("Exiting...")
            break

        else:
            if command == "1" and recording:
                print("Already recording.")
            elif command == "2" and not recording:
                print("Not currently recording.")
            else:
                print("Unknown command.")

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    if recording:
        picam2.stop_recording()
        actual_duration = time.time() - start_time
        final_filename = f"{output_folder}/{datetime.now().strftime('%y%m%d_%H%M%S')}_{round(actual_duration)}s.h264"
        shutil.move(temp_filename, final_filename)
        print(f"\nRecording stopped. Duration: {actual_duration:.1f}s. Saved as {final_filename}")

    preview.stop()
    picam2.stop()
    print("Camera resources released.")
