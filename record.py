#!/usr/bin/env python3

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from libcamera import Transform
import time
import os
import threading
import sys
import subprocess
from datetime import datetime
import shutil
import curses

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
        "Contrast": 1.0,     # 0 to 32 (1 default)
        "Saturation": 1.0,   # 0 to 32 (1 default)
        "Sharpness": 16.0    # 0 to 16 (1 default)
    },
    transform=Transform(rotation=0)  # Rotate the preview by 90 degrees
)
picam2.configure(video_config)

encoder = H264Encoder(bitrate=10000000)  # High bitrate for quality recording

picam2.start_preview(True)
picam2.start()

recording = False
camera_started = True  # Track if camera is started
temp_filename = ""
start_time = 0
stop_thread = False

# Function to display recording duration
def display_duration():
    while recording and not stop_thread:
        elapsed = time.time() - start_time
        sys.stdout.write(f"\rRecording duration: {elapsed:.1f}s")
        sys.stdout.flush()
        time.sleep(0.1)

# Function to convert H264 to MP4
def convert_to_mp4(h264_path, mp4_path, fps=30):
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

# Function to handle curses UI
def main(stdscr):
    global recording, stop_thread, temp_filename, start_time

    # Initialize curses
    curses.curs_set(0)  # Hide the cursor
    stdscr.clear()
    stdscr.refresh()

    # Menu options
    menu = ["Start Recording", "Stop Recording", "Quit"]
    current_row = 0

    def print_menu():
        stdscr.clear()
        stdscr.addstr(0, 0, "Recording System (720p@30fps, Preview ON)", curses.A_BOLD)
        stdscr.addstr(1, 0, "Use arrow keys to navigate and press Enter to select.", curses.A_DIM)
        for idx, row in enumerate(menu):
            if idx == current_row:
                stdscr.addstr(idx + 3, 0, f"> {row}", curses.A_REVERSE)  # Highlight the selected option
            else:
                stdscr.addstr(idx + 3, 0, f"  {row}")
        stdscr.refresh()

    # Main loop
    while True:
        print_menu()
        key = stdscr.getch()

        # Navigate menu
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(menu) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:  # Enter key
            if current_row == 0:  # Start Recording
                if not recording:
                    temp_filename = f"{output_folder}/temp_recording.h264"
                    picam2.start_recording(encoder, FileOutput(temp_filename))
                    recording = True
                    stop_thread = False
                    start_time = time.time()
                    stdscr.addstr(len(menu) + 5, 0, "Recording started...", curses.A_BOLD)
                    stdscr.refresh()

                    # Start duration thread
                    duration_thread = threading.Thread(target=display_duration, daemon=True)
                    duration_thread.start()
                else:
                    stdscr.addstr(len(menu) + 5, 0, "Already recording!", curses.A_BOLD)
                    stdscr.refresh()

            elif current_row == 1:  # Stop Recording
                if recording:
                    try:
                        picam2.stop_recording()
                    except Exception:
                        pass
                    recording = False
                    stop_thread = True

                    # Save recording
                    actual_duration = time.time() - start_time
                    seconds = int(actual_duration)
                    tenths = int((actual_duration - seconds) * 10)
                    duration_str = f"{seconds}_{tenths}s"

                    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
                    final_filename = f"{output_folder}/{timestamp}_{duration_str}.h264"
                    final_mp4 = f"{output_folder}/{timestamp}_{duration_str}.mp4"

                    shutil.move(temp_filename, final_filename)
                    stdscr.addstr(len(menu) + 5, 0, f"Saved as {final_filename}", curses.A_BOLD)
                    stdscr.refresh()

                    # Convert to mp4
                    convert_to_mp4(final_filename, final_mp4, fps=30)
                else:
                    stdscr.addstr(len(menu) + 5, 0, "Not currently recording!", curses.A_BOLD)
                    stdscr.refresh()

            elif current_row == 2:  # Quit
                if recording:
                    try:
                        picam2.stop_recording()
                    except Exception:
                        pass
                    recording = False
                    stop_thread = True
                break

        stdscr.refresh()

# Run curses application
try:
    curses.wrapper(main)
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
