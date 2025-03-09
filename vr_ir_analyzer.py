from picamera2 import Picamera2
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import skew
import pandas as pd
import os
import threading

class VRControllerIRAnalyzer:
    def __init__(self):
        # Configuration parameters
        self.window_size = 30  # frames for analysis window
        self.typing_skew_threshold = 0.8  # initial threshold
        self.output_dir = "ir_data_" + time.strftime("%Y%m%d-%H%M%S")
        self.recording = False
        self.calibration_mode = False
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Data storage
        self.signal_window = []
        self.timestamps = []
        self.typing_events = []
        self.skewness_values = []
        
        # Initialize the camera
        self.setup_camera()
    
    def setup_camera(self):
        """Set up and configure the Raspberry Pi NoIR camera"""
        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(main={"size": (1280, 720)})
        self.picam2.configure(self.config)
        self.picam2.start()
        
        # Adjust exposure for better IR visibility
        self.picam2.set_controls({"ExposureTime": 8000, "AnalogueGain": 8.0})
        print("Camera initialized successfully")
    
    def process_frame(self, frame):
        """Extract IR signal intensity and features from a frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to isolate IR LEDs
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Count white pixels (IR LED signals)
        signal_strength = np.sum(thresh) / 255
        
        # Find contours (IR LED positions)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate centroid positions for LEDs
        centroids = []
        for contour in contours:
            if cv2.contourArea(contour) > 5:  # Filter small noise
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centroids.append((cX, cY))
        
        return {
            'signal_strength': signal_strength,
            'contour_count': len(contours),
            'centroids': centroids,
            'frame': frame,
            'thresh': thresh
        }
    
    def analyze_fluctuations(self, signal_window):
        """Analyze signal fluctuations to detect typing events"""
        if len(signal_window) < self.window_size:
            return None, 0
        
        # Calculate differences between consecutive signal values
        diffs = np.diff([s['signal_strength'] for s in signal_window])
        
        # Calculate skewness of the differences
        skewness = skew(diffs)
        
        # Calculate variance as additional feature
        variance = np.var(diffs)
        
        return skewness, variance
    
    def identify_keystroke_pattern(self, signal_window):
        """Attempt to identify specific keystroke patterns"""
        # Extract features from the signal window
        strengths = [s['signal_strength'] for s in signal_window]
        counts = [s['contour_count'] for s in signal_window]
        
        # Calculate pattern features
        if not strengths:
            return "No signal data"
        
        peak_strength = max(strengths)
        valley_strength = min(strengths)
        strength_range = peak_strength - valley_strength
        
        # Calculate timing features
        try:
            peak_idx = strengths.index(peak_strength) 
            rise_time = peak_idx / len(strengths)
        except:
            rise_time = 0.5
        
        # Simple pattern matching (would need calibration)
        if strength_range > 1000 and rise_time < 0.3:
            return "Quick tap - possibly spacebar or enter"
        elif strength_range > 500 and rise_time > 0.5:
            return "Slow press - possibly shift or function key"
        elif np.mean(counts) > 10:
            return "Multiple LEDs active - possibly multiple keys or gesture"
        else:
            return "Unknown keystroke pattern"
    
    def calibrate_system(self):
        """Calibrate the system by collecting baseline and typing samples"""
        print("CALIBRATION MODE")
        print("1. Keep controllers still for 5 seconds")
        
        # Collect baseline (no typing) data
        baseline_data = []
        start_time = time.time()
        
        while time.time() - start_time < 5:
            frame = self.picam2.capture_array()
            signal_info = self.process_frame(frame)
            baseline_data.append(signal_info['signal_strength'])
            
            cv2.putText(frame, "CALIBRATING: KEEP STILL", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1)
        
        print("2. Now type on the virtual keyboard for 10 seconds")
        
        # Collect typing data
        typing_data = []
        start_time = time.time()
        
        while time.time() - start_time < 10:
            frame = self.picam2.capture_array()
            signal_info = self.process_frame(frame)
            typing_data.append(signal_info['signal_strength'])
            
            cv2.putText(frame, "CALIBRATING: PLEASE TYPE", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1)
        
        # Calculate optimal threshold based on collected data
        if len(baseline_data) > 1 and len(typing_data) > 1:
            baseline_diffs = np.diff(baseline_data)
            typing_diffs = np.diff(typing_data)
            
            baseline_skew = skew(baseline_diffs)
            typing_skew = skew(typing_diffs)
            
            # Set threshold between baseline and typing skewness
            optimal_threshold = (abs(baseline_skew) + abs(typing_skew)) / 2
            
            print(f"Calibration complete!")
            print(f"Baseline skewness: {baseline_skew:.2f}")
            print(f"Typing skewness: {typing_skew:.2f}")
            print(f"Optimal threshold: {optimal_threshold:.2f}")
            
            self.typing_skew_threshold = optimal_threshold
            
            # Save calibration data
            pd.DataFrame({
                'baseline_data': baseline_data,
                'typing_data': typing_data[:len(baseline_data)] if len(typing_data) > len(baseline_data) else typing_data
            }).to_csv(f"{self.output_dir}/calibration_data.csv", index=False)
            
            return optimal_threshold
        else:
            print("Insufficient data for calibration")
            return self.typing_skew_threshold
    
    def update_visualization(self):
        """Periodically update visualization of skewness distribution"""
        while self.recording:
            if len(self.skewness_values) > 30:
                plt.figure(figsize=(10, 6))
                plt.hist(self.skewness_values, bins=30, alpha=0.7)
                plt.title("Density Distribution of Skewness in IR Signal Fluctuations")
                plt.xlabel("Skewness")
                plt.ylabel("Frequency")
                plt.axvline(self.typing_skew_threshold, color='r', linestyle='--', 
                          label=f"Typing Threshold ({self.typing_skew_threshold:.2f})")
                plt.axvline(-self.typing_skew_threshold, color='r', linestyle='--')
                plt.legend()
                plt.savefig(f"{self.output_dir}/skewness_distribution_{int(time.time())}.png")
                plt.close()
            time.sleep(10)
    
    def start_recording(self):
        """Start recording and analyzing IR signals"""
        self.recording = True
        self.signal_window = []
        self.timestamps = []
        self.typing_events = []
        self.skewness_values = []
        
        # Start visualization thread
        viz_thread = threading.Thread(target=self.update_visualization)
        viz_thread.daemon = True
        viz_thread.start()
        
        try:
            print("Starting IR signal capture. Press Ctrl+C to exit.")
            
            while self.recording:
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Process frame to extract signal data
                signal_info = self.process_frame(frame)
                current_time = time.time()
                
                # Add to signal window
                self.signal_window.append(signal_info)
                self.timestamps.append(current_time)
                
                # Keep window at fixed size
                if len(self.signal_window) > self.window_size:
                    self.signal_window.pop(0)
                    self.timestamps.pop(0)
                
                # Analyze fluctuations if we have enough data
                if len(self.signal_window) == self.window_size:
                    skewness, variance = self.analyze_fluctuations(self.signal_window)
                    
                    if skewness is not None:
                        self.skewness_values.append(skewness)
                        
                        # Detect typing event based on skewness threshold
                        is_typing = abs(skewness) > self.typing_skew_threshold
                        
                        if is_typing:
                            self.typing_events.append(current_time)
                            keystroke_pattern = self.identify_keystroke_pattern(self.signal_window)
                            print(f"Typing event detected! Skewness: {skewness:.2f}, Variance: {variance:.2f}")
                            print(f"Pattern: {keystroke_pattern}")
                
                # Display the processed frame
                display_frame = signal_info['frame'].copy()
                
                cv2.putText(display_frame, 
                          f"Signal: {signal_info['signal_strength']:.0f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw centroids on frame
                for centroid in signal_info['centroids']:
                    cv2.circle(display_frame, centroid, 5, (0, 255, 0), -1)
                
                # Add typing indicator if recent typing event
                if self.typing_events and (current_time - self.typing_events[-1]) < 1.0:
                    cv2.putText(display_frame, "TYPING DETECTED", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show frames
                cv2.imshow("IR View", display_frame)
                cv2.imshow("Thresholded IR", signal_info['thresh'])
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Stopping capture...")
        finally:
            self.recording = False
            self.save_data()
    
    def save_data(self):
        """Save captured data for further analysis"""
        if self.signal_window:
            # Extract data from signal window
            df = pd.DataFrame({
                'timestamp': self.timestamps,
                'signal_strength': [s['signal_strength'] for s in self.signal_window],
                'contour_count': [s['contour_count'] for s in self.signal_window]
            })
            
            # Save skewness values
            skew_df = pd.DataFrame({'skewness': self.skewness_values})
            
            # Save typing events
            events_df = pd.DataFrame({'event_time': self.typing_events})
            
            # Save all data to CSV
            df.to_csv(f"{self.output_dir}/ir_signal_data.csv", index=False)
            skew_df.to_csv(f"{self.output_dir}/skewness_values.csv", index=False)
            events_df.to_csv(f"{self.output_dir}/typing_events.csv", index=False)
            
            print(f"Data saved to {self.output_dir}/")
    
    def cleanup(self):
        """Clean up resources"""
        cv2.destroyAllWindows()
        self.picam2.stop()
        print("Resources cleaned up")

# Main execution
if __name__ == "__main__":
    analyzer = VRControllerIRAnalyzer()
    
    # Menu system
    while True:
        print("\n=== VR Controller IR Signal Analyzer ===")
        print("1. Calibrate System")
        print("2. Start Recording")
        print("3. Exit")
        
        choice = input("Select an option: ")
        
        if choice == '1':
            analyzer.calibrate_system()
        elif choice == '2':
            analyzer.start_recording()
        elif choice == '3':
            analyzer.cleanup()
            break
        else:
            print("Invalid option, please try again")
