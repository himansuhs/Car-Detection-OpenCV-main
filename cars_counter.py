# Importing libraries
import cv2
import numpy as np

# Capture video
cap = cv2.VideoCapture('traffic.mp4')  # Change 'traffic.mp4' to any video path

# Get the video's frame width, height, and fps
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate a dynamic line height based on video height
line_height = int(frame_height * 0.8)  # Set counting line 80% down the video height

# Minimum contour width and height, dynamically adjusted based on frame size
min_contour_width = frame_width // 30  # Approx 1/30th of the frame width
min_contour_height = frame_height // 30  # Approx 1/30th of the frame height
offset = 10  # Offset for counting line

# Initialize variables
cars = 0
matches = []

# Define a function to calculate the centroid of a bounding box
def get_centroid(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy

# Set video resolution if needed (optional for high-resolution)
cap.set(3, frame_width)
cap.set(4, frame_height)

# Read the first two frames
if cap.isOpened():
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
else:
    ret = False

while ret:
    # Calculate the absolute difference between the two frames
    d = cv2.absdiff(frame1, frame2)

    # Convert the difference to grayscale
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    # Apply thresholding to isolate significant differences
    ret, th = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill small gaps
    dilated = cv2.dilate(th, np.ones((3, 3)))

    # Morphological closing to fill small holes in the detected areas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary image
    contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over all contours
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)

        # Check if the contour meets the minimum size criteria
        if w >= min_contour_width and h >= min_contour_height:
            # Draw a rectangle around the detected object
            cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

            # Draw the counting line
            cv2.line(frame1, (0, line_height), (frame_width, line_height), (0, 255, 0), 2)

            # Get the centroid of the contour
            centroid = get_centroid(x, y, w, h)
            matches.append(centroid)

            # Draw a circle at the centroid
            cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

            # Check if the centroid has crossed the counting line
            cx, cy = centroid
            for (x, y) in matches:
                if (line_height + offset) > y > (line_height - offset):
                    cars += 1
                    matches.remove((x, y))
                    print(f"Car Count: {cars}")

    # Display the car count on the video
    cv2.putText(frame1, f"Total Vehicles Detected: {cars}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)

    # Show the processed frame
    cv2.imshow("Vehicle Detection", frame1)

    # Break the loop if 'ESC' key is pressed
    if cv2.waitKey(1) == 27:
        break

    # Move to the next frame
    frame1 = frame2
    ret, frame2 = cap.read()

# Release resources and close windows
cv2.destroyAllWindows()
cap.release()
