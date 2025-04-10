#!/usr/bin/env python3

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import time
import os
from datetime import datetime
import shutil

# Create output folder
output_folder = "recordings"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize camera with optimized settings for smoother video
picam2 = Picamera2()

# Lower resolution configuration (720p instead of 1080p)
video_config = picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    lores={"size": (640, 480), "format": "YUV420"},
    display="lores",
    encode="lores"
)

# Set specific framerate (30fps is usually smoother than higher rates)
video_config["controls"]["FrameRate"] = 30.0 [[3]]

picam2.configure(video_config)

# Reduce bitrate for smoother encoding (2 Mbps instead of 10 Mbps)
encoder = H264Encoder(bitrate=2000000)

# Start the camera with preview
picam2.start_preview(True)
picam2.start()

recording = False
temp_filename = ""
start_time = 0

print("Flexible Recording System Ready (Optimized for smooth recording)")
print("Commands:")
print("  s - Start/Resume recording")
print("  p - Pause recording")
print("  q - Quit")

try:
    while True:
        command = input("> ")
        
        if command.lower() == 's' and not recording:
            # Use a temporary filename during recording
            temp_filename = f"{output_folder}/temp_recording.h264"
            
            # Start recording (preview continues)
            picam2.start_recording(encoder, FileOutput(temp_filename))
            recording = True
            start_time = time.time()
            print(f"Recording started...")
            
        elif command.lower() == 'p' and recording:
            # Stop the current recording
            picam2.stop_recording()
            recording = False
            
            # Calculate recording length in seconds (rounded)
            duration = round(time.time() - start_time)
            
            # Create the final filename with date and length
            date_part = datetime.now().strftime("%y%m%d")
            final_filename = f"{output_folder}/{date_part}_{duration}.h264"
            
            # Rename the file
            shutil.move(temp_filename, final_filename)
            print(f"Recording stopped. Saved as {final_filename}")
            
        elif command.lower() == 'q':
            # Stop recording if active and exit
            if recording:
                picam2.stop_recording()
                duration = round(time.time() - start_time)
                date_part = datetime.now().strftime("%y%m%d")
                final_filename = f"{output_folder}/{date_part}_{duration}.h264"
                shutil.move(temp_filename, final_filename)
                print(f"Recording saved as {final_filename}")
            print("Exiting...")
            break
            
        else:
            if command.lower() == 's' and recording:
                print("Already recording")
            elif command.lower() == 'p' and not recording:
                print("Not currently recording")
            else:
                print("Unknown command")
                
except KeyboardInterrupt:
    print("\nProgram interrupted")
finally:
    # Clean up
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
