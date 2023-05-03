import cv2
import os
from utils.dir_utils import get_next_number
# Define the function to capture the image when mouse is clicked
def capture_image(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        dataset_path = './dataset/dataset_rover2/images'
        next_number = get_next_number(dataset_path)
        cv2.imwrite(os.path.join(dataset_path,f'{next_number}.png'), frame)
        print(f"Data saved #{next_number}")
        # Save the current frame as an image
        cv2.imwrite("captured_image.jpg", frame)
        print("Image captured!")


# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Unable to open camera")
    exit()

# Create a window to display the camera feed
cv2.namedWindow("Camera Feed")

# Set the mouse callback function to capture the image
cv2.setMouseCallback("Camera Feed", capture_image)

# Start the camera feed
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Check if frame is available
    if not ret:
        print("Unable to capture frame")
        break

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    # Check for key events
    key = cv2.waitKey(1)
    if key == ord('q'):
        # Quit the program if 'q' key is pressed
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
