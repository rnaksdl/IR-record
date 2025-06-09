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

# Print available sensor modes for reference
print("Available sensor modes:")
for idx, mode in enumerate(picam2.sensor_modes):
    print(f"Mode {idx}: {mode}")

# Use 1080p47 (1920x1080 at 47 FPS), the highest supported 16:9 mode
video_config = picam2.create_video_configuration(
    main={"size": (1640, 1232)},
    controls={
        "FrameRate": 60.0,
        "ExposureTime": 20000,        # 20ms exposure
        "AnalogueGain": 8.0,          # Increased gain
        "Contrast": 1.5,              # Higher contrast
        "Brightness": -0.2,           # Reduced brightness
        "Saturation": 1.2,            # Enhanced color
        "Sharpness": 0.0,             # Minimal sharpening
        "AeEnable": False,            # Manual exposure
        "AwbEnable": False            # Manual white balance
    },
    transform=Transform(hflip=1)
)
picam2.configure(video_config)

encoder = H264Encoder(bitrate=10000000)  # Higher bitrate for 1080p

picam2.start_preview(True)
picam2.start()

recording = False
temp_filename = ""
start_time = 0
stop_thread = False

def display_duration():
    while recording and not stop_thread:
        elapsed = time.time() - start_time
        sys.stdout.write(f"\rRecording duration: {elapsed:.1f}s")
        sys.stdout.flush()
        time.sleep(0.1)

def convert_to_mp4(h264_path, mp4_path, fps=47):
    print(f"Converting {h264_path} to {mp4_path} using ffmpeg...")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", h264_path,
        "-c", "copy",
        mp4_path
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Conversion complete: {mp4_path}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg conversion failed: {e}")

def detect_ir_lights(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for purple IR lights
    lower_purple = np.array([130, 50, 200])  # Adjust these values as needed
    upper_purple = np.array([160, 255, 255])
    
    # Create mask for purple colors
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Apply additional brightness threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, brightness_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Combine masks
    final_mask = cv2.bitwise_and(mask, brightness_mask)
    
    # Find contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    ir_lights = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 10 < area < 200:  # Adjust these thresholds based on your IR light size
            ir_lights.append(contour)
    
    return ir_lights

def process_frame(frame):
    ir_lights = detect_ir_lights(frame)
    
    # Draw detected IR lights
    for contour in ir_lights:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame

print("IR Signal Analysis Recording System (1080p47, Preview ON)")
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
            convert_to_mp4(final_filename, final_mp4, fps=47)
            
        elif command == "3":
            if recording:
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
                print(f"Recording saved as {final_filename}")

                # Convert to mp4
                convert_to_mp4(final_filename, final_mp4, fps=47)
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
        sys.stdout.write("\n")
        picam2.stop_recording()
        
        actual_duration = time.time() - start_time
        seconds = int(actual_duration)
        tenths = int((actual_duration - seconds) * 10)
        duration_str = f"{seconds}_{tenths}s"
        
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        final_filename = f"{output_folder}/{timestamp}_{duration_str}.h264"
        final_mp4 = f"{output_folder}/{timestamp}_{duration_str}.mp4"
        
        shutil.move(temp_filename, final_filename)
        print(f"Recording saved as {final_filename}")

        # Convert to mp4
        convert_to_mp4(final_filename, final_mp4, fps=47)

    try:
        picam2.stop_preview()
    except Exception:
        pass
    try:
        picam2.stop()
    except Exception:
        pass

    print("Camera resources released")
