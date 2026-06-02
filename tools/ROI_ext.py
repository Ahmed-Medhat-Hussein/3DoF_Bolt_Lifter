import cv2

# Global variables to track mouse state
ref_point = []
cropping = False

def shape_selection(event, x, y, flags, param):
    global ref_point, cropping

    # Record starting (x, y) coordinates on left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    # Record ending (x, y) coordinates on release
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        # Draw the final rectangle
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("ROI Selector", image)

# Initialize camera
cap = cv2.VideoCapture(0) # Change index if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Instructions:")
print("1. Click and drag to select a region.")
print("2. Press 'c' to crop and view.")
print("3. Press 'r' to reset.")
print("4. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    image = frame.copy()
    cv2.namedWindow("ROI Selector")
    cv2.setMouseCallback("ROI Selector", shape_selection)

    while True:
        # Display the image and wait for a keypress
        cv2.imshow("ROI Selector", image)
        key = cv2.waitKey(1) & 0xFF

        # Reset the ROI
        if key == ord("r"):
            image = frame.copy()
            ref_point = []

        # If 'c' is pressed, crop the ROI and show it
        elif key == ord("c"):
            if len(ref_point) == 2:
                # Calculate coordinates
                x1, y1 = ref_point[0]
                x2, y2 = ref_point[1]
                roi = frame[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
                
                print(f"ROI Coordinates: X={min(x1,x2)}, Y={min(y1,y2)}, W={abs(x1-x2)}, H={abs(y1-y2)}")
                cv2.imshow("Cropped ROI", roi)
                cv2.waitKey(0)
                cv2.destroyWindow("Cropped ROI")

        # Break loop to get a fresh frame (live preview)
        elif key == ord(" ") or key == ord("q"):
            break

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()