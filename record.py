#!/usr/bin/env python3

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from libcamera import Transform
import time
import os
from datetime import datetime
import shutil

# Create output folder
output_folder = "recordings"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize camera with focus on high frame rate
picam2 = Picamera2()

# Configuration with horizontal flip and high frame rate
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

print("IR Signal Analysis Recording System")
print("Commands:")
print("  1 - Start recording")
print("  2 - Stop recording")
print("  3 - Quit")

try:
    while True:
        # Show recording duration in prompt if active
        if recording:
            elapsed = time.time() - start_time
            command = input(f"[Recording: {elapsed:.1f}s] > ")
        else:
            command = input("> ")
        
        if command == "1" and not recording:
            temp_filename = f"{output_folder}/temp_recording.h264"
            picam2.start_recording(encoder, FileOutput(temp_filename))
            recording = True
            start_time = time.time()
            print(f"Recording started at {datetime.now().strftime('%H:%M:%S')}")
            
        elif command == "2" and recording:
            picam2.stop_recording()
            actual_duration = time.time() - start_time
            recording = False
            
            duration = round(actual_duration)
            date_part = datetime.now().strftime("%y%m%d")
            final_filename = f"{output_folder}/{date_part}_{duration}.h264"
            shutil.move(temp_filename, final_filename)
            print(f"\nRecording stopped after {actual_duration:.1f} seconds. Saved as {final_filename}")
            
        elif command == "3":
            if recording:
                actual_duration = time.time() - start_time
                duration = round(actual_duration)
                picam2.stop_recording()
                date_part = datetime.now().strftime("%y%m%d")
                final_filename = f"{output_folder}/{date_part}_{duration}.h264"
                shutil.move(temp_filename, final_filename)
                print(f"\nRecording stopped after {actual_duration:.1f} seconds. Saved as {final_filename}")
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
        actual_duration = time.time() - start_time
        duration = round(actual_duration)
        picam2.stop_recording()
        date_part = datetime.now().strftime("%y%m%d")
        final_filename = f"{output_folder}/{date_part}_{duration}.h264"
        shutil.move(temp_filename, final_filename)
        print(f"\nRecording stopped after {actual_duration:.1f} seconds. Saved as {final_filename}")
    picam2.stop_preview()
    picam2.stop()
    print("Camera resources released")
