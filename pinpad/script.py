import cv2

# Load your image
img = cv2.imread('pinpad.jpg')  # <-- Change filename if needed
clone = img.copy()
key_labels = ['1','2','3','4','5','6','7','8','9','0']
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = len(points)
        if idx < len(key_labels):
            print(f"Key {key_labels[idx]}: ({x}, {y})")
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, key_labels[idx], (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('image', img)
        if len(points) == len(key_labels):
            print("All key centers recorded:")
            for label, pt in zip(key_labels, points):
                print(f"{label}: {pt}")

cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
print("Click the center of each key in order: 1 2 3 4 5 6 7 8 9 0")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save to file if you want
import numpy as np
np.savetxt('pinpad_key_centers.txt', points, fmt='%d')