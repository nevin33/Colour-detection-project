
import cv2
import numpy as np

def process_frame(frame):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, blue, and green
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    blue_lower = np.array([110, 50, 50])
    blue_upper = np.array([130, 255, 255])
    green_lower = np.array([50, 100, 100])
    green_upper = np.array([70, 255, 255])

    # Create masks for each color
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)
    blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

    # Find contours in each color mask
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each color
    process_color(frame, contours_red, (0, 0, 255), "Red")
    process_color(frame, contours_blue, (255, 0, 0), "Blue")
    process_color(frame, contours_green, (0, 255, 0), "Green")

def process_color(frame, contours, color, label):
    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) > 100:
            # Draw bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Calculate coordinates of the center
            center_x = x + w // 2
            center_y = y + h // 2

            # Display coordinates
            cv2.putText(frame, f"{label}: ({center_x}, {center_y})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            print("Error reading frame")
            break

        # Process the frame
        process_frame(frame)

        # Display the result
        cv2.imshow('Color Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Call the main function
    main()