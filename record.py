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
import subprocess
import cv2
import numpy as np

# Settings
output_folder = "recordings"
processed_folder = "processed"
record_width = 1440
record_height = 1080
record_fps = 30.0  # Standard, universally compatible
bitrate = 10000000

# Create necessary directories
os.makedirs(output_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)

picam2 = Picamera2()

# Print available sensor modes for reference
print("Available sensor modes:")
for idx, mode in enumerate(picam2.sensor_modes):
    print(f"Mode {idx}: {mode}")

video_config = picam2.create_video_configuration(
    main={"size": (record_width, record_height)},
    controls={"FrameRate": record_fps},
    transform=Transform(hflip=1)
)
picam2.configure(video_config)

encoder = H264Encoder(bitrate=bitrate)

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

def convert_to_mp4(h264_path, mp4_path, fps):
    print(f"Converting {h264_path} to {mp4_path} using ffmpeg at {fps} FPS...")
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
    """Detect purple IR lights in a frame and return their bounding boxes"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # need to find correct value for this
    lower_purple = np.array([100, 20, 20])
    upper_purple = np.array([170, 255, 255])
    
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contours found: {len(contours)}")
    
    return [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 20]

def calculate_crop_region(video_path):
    """Calculate minimum crop region containing all IR lights throughout the video"""
    cap = cv2.VideoCapture(video_path)
    x_min = y_min = float('inf')
    x_max = y_max = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        boxes = detect_ir_lights(frame)
        for (x, y, w, h) in boxes:
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
    
    cap.release()
    
    # Add 10% padding and clamp to video dimensions
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pad_x = int((x_max - x_min) * 0.1)
    pad_y = int((y_max - y_min) * 0.1)
    
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(width, x_max + pad_x)
    y_max = min(height, y_max + pad_y)
    
    return x_min, y_min, x_max - x_min, y_max - y_min

def process_video(input_path):
    """Process video to crop to IR-active regions"""
    crop_params = calculate_crop_region(input_path)
    
    if crop_params[2] == 0 or crop_params[3] == 0:
        raise ValueError("No IR lights detected in the video")
    
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    output_path = f"{processed_folder}/{timestamp}_cropped.mp4"
    
    cmd = [
        'ffmpeg', '-i', input_path,
        '-filter:v', f'crop={crop_params[2]}:{crop_params[3]}:{crop_params[0]}:{crop_params[1]}',
        '-c:a', 'copy', output_path
    ]
    
    subprocess.run(cmd, check=True)
    return output_path

def processing_menu():
    print("\nVideo Processing Options:")
    print("  1 - Process recorded video")
    print("  2 - Return to main menu")
    
    while True:
        cmd = input("Processing> ")
        
        if cmd == "1":
            input_path = input("Enter path to recorded video: ").strip()
            if not os.path.exists(input_path):
                print("File not found!")
                continue
            try:
                output_path = process_video(input_path)
                print(f"Processed video saved to: {output_path}")
            except Exception as e:
                print(f"Processing failed: {str(e)}")
        elif cmd == "2":
            break
        else:
            print("Invalid processing command")

print(f"IR Signal Analysis Recording System ({record_width}x{record_height}@{int(record_fps)}fps, Preview ON)")
print("Commands:")
print("  1 - Start recording")
print("  2 - Stop recording")
print("  3 - Quit")
print("  4 - Video processing")

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

            convert_to_mp4(final_filename, final_mp4, fps=record_fps)
            
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
                convert_to_mp4(final_filename, final_mp4, fps=record_fps)
            print("Exiting...")
            break
        
        elif command == "4":
            processing_menu()
            print("\nMain Commands:")
            print("  1 - Start recording")
            print("  2 - Stop recording")
            print("  3 - Quit")
            print("  4 - Video processing")
            
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
        convert_to_mp4(final_filename, final_mp4, fps=record_fps)
        
    try:
        picam2.stop_preview()
    except Exception as e:
        print(f"Error stopping preview: {e}")
    try:
        picam2.stop()
    except Exception as e:
        print(f"Error stopping camera: {e}")
        
    print("Camera resources released")
