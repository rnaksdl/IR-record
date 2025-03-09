from picamera2 import Picamera2
import cv2
import numpy as np
import time

# Initialize the camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.start()

# Optional: Adjust exposure for better IR visibility
picam2.set_controls({"ExposureTime": 10000, "AnalogueGain": 10.0})

print("Camera started. Press Ctrl+C to exit.")

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()
        
        # Optional: Apply threshold to highlight bright IR LEDs
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Display both original and thresholded views
        cv2.imshow("IR View", frame)
        cv2.imshow("Thresholded IR", thresh)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("Stopping...")
finally:
    cv2.destroyAllWindows()
    picam2.stop()
