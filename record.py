import picamera
import time
import os
from datetime import datetime

# Initialize the camera
camera = picamera.PiCamera()
camera.resolution = (1920, 1080)  # Full HD resolution
camera.framerate = 30

recording = False
segment_count = 0
output_folder = "recordings"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
            filename = f"{output_folder}/segment_{timestamp}.h264"
            
            # Start recording
            camera.start_recording(filename)
            recording = True
            segment_count += 1
            print(f"Recording started (segment {segment_count})")
            
        elif command.lower() == 'p' and recording:
            # Stop the current recording
            camera.stop_recording()
            recording = False
            print("Recording paused")
            
        elif command.lower() == 'q':
            # Stop recording if active and exit
            if recording:
                camera.stop_recording()
                print("Recording stopped")
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
        camera.stop_recording()
    camera.close()
    print("Camera resources released")
