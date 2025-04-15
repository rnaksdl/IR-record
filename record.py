#!/usr/bin/env python3

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from libcamera import Transform
import time
import os
from datetime import datetime
import shutil

output_folder = "recordings"
os.makedirs(output_folder, exist_ok=True)  # Improved directory creation

# Initialize camera with correct preview sequence
picam2 = Picamera2()
config = picam2.create_video_configuration(
    transform=Transform(hflip=1),
    controls={"FrameRate": 60.0}         # Flip first, then configure
)
picam2.configure(config)

# Start camera before preview to maintain stream
picam2.start()  # CAMERA MUST START BEFORE PREVIEW
preview = picam2.start_preview()  # Non-blocking preview  [[9]]
encoder = H264Encoder(bitrate=4000000)

recording = False
temp_filename = ""
start_time = 0
last_update = 0  # For time display updates

def update_console(cmd=False):
    global last_update
    now = time.time()
    if recording and (now - last_update > 1 or cmd):
        elapsed = now - start_time
        print(f"\rRecording: {elapsed:.1f}s", end='')
        last_update = now

print("IR Signal Analysis Recording System")
print("  [1] Start    [2] Stop    [3] Exit")

try:
    while True:
        update_console()
        
        # Non-blocking input check using select
        import select, sys
        if select.select([sys.stdin], [], [], 0)[0]:
            command = sys.stdin.readline().strip()
            if command == "1" and not recording:
                temp_filename = f"{output_folder}/temp_recording.h264"
                picam2.start_recording(encoder, FileOutput(temp_filename))
                start_time = time.time()
                recording = True
                print(f"\nStarted recording at {datetime.now().strftime('%H:%M:%S')}")

            elif command == "2" and recording:
                actual_duration = time.time() - start_time
                picam2.stop_recording()
                recording = False
                update_console(True)  # Show stopped recording time
                final_filename = f"{output_folder}/{datetime.now().strftime('%y%m%d')}_{round(actual_duration)}.h264"
                shutil.move(temp_filename, final_filename)
                print(f"\nSaved {actual_duration:.1f}s -> {final_filename}")

            elif command == "3":
                if recording:
                    picam2.stop_recording()
                    actual_duration = time.time() - start_time
                    final_filename = f"{output_folder}/{datetime.now().strftime('%y%m%d')}_{round(actual_duration)}.h264"
                    shutil.move(temp_filename, final_filename)
                    print(f"\nSaved {actual_duration:.1f}s -> {final_filename}")
                break

            elif command:
                status_msg = {
                    "1": "Already recording!",
                    "2": "No active recording",
                    "3": ""
                }.get(command, "Invalid command")
                if status_msg:
                    print(f"\n{status_msg}")

except KeyboardInterrupt:
    print("\nProgram halted")
    
finally:
    if recording:
        picam2.stop_recording()
    picam2.stop_preview()
    picam2.stop()
    print("Camera resources released")
