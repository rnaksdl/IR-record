#!/usr/bin/env python3

import os
import sys
import time
import shutil
import threading
from datetime import datetime
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from libcamera import Transform

# --- System Optimizations ---
def optimize_system():
    try:
        os.nice(-20)
    except Exception as e:
        print(f"Warning: Could not set process priority: {e}")

    try:
        os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
    except Exception as e:
        print(f"Warning: Could not set CPU governor: {e}")

    os.system('sysctl -w net.core.rmem_max=12582912')
    os.system('sysctl -w net.core.wmem_max=12582912')

# --- Camera Configuration ---
def configure_camera():
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "YUV420"},
        controls={
            "FrameRate": 60.0,
            "NoiseReductionMode": 0,
            "AwbEnable": 0,
            "AeEnable": 0,
        },
        buffer_count=4,
        transform=Transform(hflip=1)
    )
    picam2.configure(video_config)
    return picam2

# --- Encoder Configuration ---
def configure_encoder():
    encoder = H264Encoder(
        bitrate=4000000,
        profile="baseline",
        repeat=False,
        iperiod=0
    )
    return encoder

# --- Main Script ---
def main():
    optimize_system()

    output_folder = "recordings"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    picam2 = configure_camera()
    encoder = configure_encoder()

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

    print("IR Signal Analysis Recording System (Optimized for Max FPS)")
    print("Commands:")
    print("  1 - Start recording")
    print("  2 - Stop recording")
    print("  3 - Quit")

    try:
        while True:
            command = input("> ")

            if command == "1" and not recording:
                temp_filename = f"{output_folder}/temp_recording.h264"
                os.sync()
                os.system('sync; echo 3 > /proc/sys/vm/drop_caches')
                picam2.start_recording(encoder, FileOutput(temp_filename, threads=4))
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
                shutil.move(temp_filename, final_filename)
                print(f"Saved as {final_filename}")

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
            sys.stdout.write("\n")
            picam2.stop_recording()
            actual_duration = time.time() - start_time
            seconds = int(actual_duration)
            tenths = int((actual_duration - seconds) * 10)
            duration_str = f"{seconds}_{tenths}s"
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            final_filename = f"{output_folder}/{timestamp}_{duration_str}.h264"
            shutil.move(temp_filename, final_filename)
            print(f"Recording saved as {final_filename}")
        picam2.stop_preview()
        picam2.stop()
        print("Camera resources released")

if __name__ == "__main__":
    main()
