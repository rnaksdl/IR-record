#!/usr/bin/env python3

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from libcamera import Transform
import time
import os
import cv2
import numpy as np
import threading
import sys
import subprocess
from datetime import datetime
import shutil

# Create output folder
output_folder = "recordings"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

picam2 = Picamera2()

# Camera configuration with reduced brightness and increased contrast
video_config = picam2.create_video_configuration(
    main={"size": (1280, 720)},  # Lower resolution for testing
    controls={
        "Brightness": -1.0,  # -1 to 1 (0 default)
        "Contrast": 1.0, # 0 to 32 (1 default)
        "Saturation": 1.0, # 0 to 32 (1 default)
        "Sharpness": 16.0 # 0 to 16 (1 default)

    },
    transform=Transform(hflip=1)
)
picam2.configure(video_config)

encoder = H264Encoder(bitrate=10000000)  # High bitrate for quality recording

picam2.start_preview(True)
picam2.start()

recording = False
camera_started = True
temp_filename = ""
start_time = 0
stop_thread = False

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

            duration_thread = threading.Thread(target=display_duration, daemon=True)
            duration_thread.start()
            
        elif command == "2" and recording:
            sys.stdout.write("\n")
            picam2.stop_recording()
            recording = False
            stop_thread = True
            
            actual_duration = time.time() - start_time
            seconds = int(actual_duration)
            tenths = int((actual_duration - seconds) * 10)
            duration_str = f"{seconds}_{tenths}s"
            
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            final_filename = f"{output_folder}/{timestamp}_{duration_str}.h264"
            final_mp4 = f"{output_folder}/{timestamp}_{duration_str}.mp4"
            
            shutil.move(temp_filename, final_filename)
            print(f"Saved as {final_filename}")

            # Convert to mp4
            convert_to_mp4(final_filename, final_mp4, fps=30)
            
        elif command == "3":
            if recording:
                sys.stdout.write("\n")
                try:
                    picam2.stop_recording()
                except Exception:
                    pass
                recording = False
                stop_thread = True
            print("Exiting...")
            break
            
        else:
            print("Unknown command")
                
except KeyboardInterrupt:
    print("\nProgram interrupted")
finally:
    # Only stop recording if still recording
    if recording:
        try:
            picam2.stop_recording()
        except Exception:
            pass
        recording = False

    # Only stop preview/camera if started
    if camera_started:
        try:
            picam2.stop_preview()
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass
        camera_started = False

    print("Camera resources released")
