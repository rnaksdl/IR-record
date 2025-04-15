#!/usr/bin/env python3

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from libcamera import Transform
import time
import os
from datetime import datetime
import shutil
import threading
import sys

# Create output folder
output_folder = "recordings"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize camera with high frame rate and mirrored preview
picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": 60.0},
    transform=Transform(hflip=1)
)
picam2.configure(video_config)

encoder = H264Encoder(bitrate=4000000)

# Start the camera with preview
picam2.start_preview(True)
picam2.start()

recording = False
temp_filename = ""
start_time = 0
stop_thread = False  # Flag to stop the duration thread

# Function to display recording duration in terminal
def display_duration():
    # Only update the current line, no newlines
    while recording and not stop_thread:
        elapsed = time.time() - start_time
        sys.stdout.write(f"\rRecording duration: {elapsed:.1f}s")
        sys.stdout.flush()
        time.sleep(0.1)
    # When done, add a newline so the prompt appears on a new line
    sys.stdout.write("\n")
    sys.stdout.flush()

print("IR Signal Analysis Recording System")
print("Commands:")
print("  1 - Start recording")
print("  2 - Stop recording")
print("  3 - Quit")

try:
    while True:
        command = input("> ")
        
        if command == "1" and not recording:
            temp_filename = f"{output_folder}/temp_recording.h264"
            picam2.start_recording(encoder, FileOutput(temp_filename))
            recording = True
            stop_thread = False
            start_time = time.time()
            print("Recording started...")

            # Start duration display thread
            duration_thread = threading.Thread(target=display_duration, daemon=True)
            duration_thread.start()
            
        elif command == "2" and recording:
            picam2.stop_recording()
            recording = False
            stop_thread = True  # Signal thread to stop
            
            duration = round(time.time() - start_time)
            date_part = datetime.now().strftime("%y%m%d")
            final_filename = f"{output_folder}/{date_part}_{duration}.h264"
            shutil.move(temp_filename, final_filename)
            print(f"Recording stopped. Duration: {duration}s. Saved as {final_filename}")
            
        elif command == "3":
            if recording:
                picam2.stop_recording()
                recording = False
                stop_thread = True
                
                duration = round(time.time() - start_time)
                date_part = datetime.now().strftime("%y%m%d")
                final_filename = f"{output_folder}/{date_part}_{duration}.h264"
                shutil.move(temp_filename, final_filename)
                print(f"Recording saved as {final_filename}")
            print("Exiting...")
            break
            
        else:
            if command == "1" and recording:
                print("Already recording")
            elif command == "2" and not recording:
                print("Not currently recording")
            else:
                print("Unknown command")
                
except KeyboardInterrupt:
    print("\nProgram interrupted")
finally:
    if recording:
        picam2.stop_recording()
        duration = round(time.time() - start_time)
        date_part = datetime.now().strftime("%y%m%d")
        final_filename = f"{output_folder}/{date_part}_{duration}.h264"
        shutil.move(temp_filename, final_filename)
        print(f"Recording saved as {final_filename}")
    picam2.stop_preview()
    picam2.stop()
    print("Camera resources released")
