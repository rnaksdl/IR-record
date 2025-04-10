#!/usr/bin/env python3

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import time
import os
from datetime import datetime

# Create output folder
output_folder = "recordings"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize camera
picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)
encoder = H264Encoder()

# Start the camera
picam2.start()

recording = False
print("Flexible Recording System Ready")
print("Commands:")
print("  s - Start/Resume recording")
print("  p - Pause recording")
print("  q - Quit")

try:
    while True:
        command = input("> ")
        
        if command.lower() == 's' and not recording:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_folder}/video_{timestamp}.h264"
            
            # Start recording
            picam2.start_recording(encoder, filename)
            recording = True
            print(f"Recording started to {filename}")
            
        elif command.lower() == 'p' and recording:
            # Stop the current recording
            picam2.stop_recording()
            recording = False
            print("Recording stopped")
            
        elif command.lower() == 'q':
            # Stop recording if active and exit
            if recording:
                picam2.stop_recording()
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
    picam2.stop()
    print("Camera resources released")
