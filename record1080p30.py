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

# Settings
output_folder = "recordings"
record_width = 1440
record_height = 1080
record_fps = 30.0
bitrate = 10000000

# Anti-flicker settings (microseconds)
# For 60Hz lighting (US/Canada): 16667 (1/60s), 33333 (1/30s)
# For 50Hz lighting (Europe/UK): 20000 (1/50s), 40000 (1/25s)
exposure_time_60hz = 16667  # 1/60s in microseconds
exposure_time_50hz = 20000  # 1/50s in microseconds
exposure_time = exposure_time_60hz  # Default to 60Hz
analogue_gain = 4.0  # Default gain, adjust as needed

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

picam2 = Picamera2()

# Print available sensor modes for reference
print("Available sensor modes:")
for idx, mode in enumerate(picam2.sensor_modes):
    print(f"Mode {idx}: {mode}")

video_config = picam2.create_video_configuration(
    main={"size": (record_width, record_height)},
    controls={
        "FrameRate": record_fps,
        "ExposureTime": exposure_time,
        "AnalogueGain": analogue_gain,
        "AeEnable": False  # Disable auto exposure for consistent results
    },
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

def set_exposure_settings(exposure_us, gain):
    picam2.set_controls({
        "ExposureTime": exposure_us,
        "AnalogueGain": gain
    })
    print(f"Exposure settings updated: ExposureTime={exposure_us}μs (1/{1000000/exposure_us:.1f}s), AnalogueGain={gain}x")

print(f"IR Signal Analysis Recording System ({record_width}x{record_height}@{int(record_fps)}fps, Preview ON)")
print("Anti-flicker settings:")
print(f"  ExposureTime: {exposure_time}μs (1/{1000000/exposure_time:.1f}s)")
print(f"  AnalogueGain: {analogue_gain}x")
print("Commands:")
print("  1 - Start recording")
print("  2 - Stop recording")
print("  3 - Set exposure for 60Hz lighting (1/60s)")
print("  4 - Set exposure for 50Hz lighting (1/50s)")
print("  5 - Increase gain (+0.5)")
print("  6 - Decrease gain (-0.5)")
print("  7 - Quit")

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

            # Convert to mp4 with correct FPS
            convert_to_mp4(final_filename, final_mp4, fps=record_fps)

        elif command == "3" and not recording:
            # Set exposure for 60Hz lighting (1/60s)
            exposure_time = exposure_time_60hz
            set_exposure_settings(exposure_time, analogue_gain)
            
        elif command == "4" and not recording:
            # Set exposure for 50Hz lighting (1/50s)
            exposure_time = exposure_time_50hz
            set_exposure_settings(exposure_time, analogue_gain)
            
        elif command == "5" and not recording:
            # Increase gain
            analogue_gain += 0.5
            set_exposure_settings(exposure_time, analogue_gain)
            
        elif command == "6" and not recording:
            # Decrease gain
            if analogue_gain > 0.5:
                analogue_gain -= 0.5
                set_exposure_settings(exposure_time, analogue_gain)
            else:
                print("Gain already at minimum")
            
        elif command == "7":
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

                # Convert to mp4 with correct FPS
                convert_to_mp4(final_filename, final_mp4, fps=record_fps)
            print("Exiting...")
            break
            
        else:
            if command == "1" and recording:
                print("Already recording")
            elif command == "2" and not recording:
                print("Not currently recording")
            elif command in ["3", "4", "5", "6"] and recording:
                print("Cannot change exposure settings during recording")
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

        # Convert to mp4 with correct FPS
        convert_to_mp4(final_filename, final_mp4, fps=record_fps)
    picam2.stop_preview()
    picam2.stop()
    print("Camera resources released")
