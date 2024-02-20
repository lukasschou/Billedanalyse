import time
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.util import img_as_ubyte

alpha = 0.95
T = 0.1
A = 5

def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name,img)

def capture_from_camera_and_show_images():
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # Get first image
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        exit()

    # Transform image to gray scale and then to float, so we can do some processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = img_as_float(frame_gray)

    # To keep track of frames per second
    start_time = time.time()
    n_frames = 0
    stop = False
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Transform image to gray scale and then to float, so we can do some processing
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_frame_gray = img_as_float(new_frame_gray)

        # Compute difference image
        dif_img = np.abs(new_frame_gray - frame_gray)

        # Apply the threshold to the difference image to create a binary image
        _, binary_image = cv2.threshold(dif_img, T, 255, cv2.THRESH_BINARY)

        # Convert the image data to a numpy array
        img_data = np.array(binary_image)
        # Add a new dimension to the array
        img_data_3d = np.expand_dims(img_data, axis=-1)

        # Count the number of non-background pixels
        foreground_pixels = np.count_nonzero(np.any(img_data_3d != 0, axis=-1))
        # Compute the total number of pixels in the image
        total_pixels = binary_image.shape[0] * binary_image.shape[1]
        # Compute the percentage of foreground pixels
        foreground_percentage = (foreground_pixels / total_pixels) * 100

        if foreground_percentage > A:
            print("Alarm")
            # Choose the text and position
            text = "Change Detected!"
            position = (50, 50)

            # Add the text to the image
            new_frame = cv2.putText(new_frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  
        # Keep track of frames-per-second (FPS)
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Put the FPS on the new_frame
        str_out = f"fps: {fps}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        # Adjust this to the desired zoom level
        zoom_factor = 30.0 
        # Calculate the new size of the image
        new_size = (int(new_frame.shape[1] * zoom_factor), int(new_frame.shape[0] * zoom_factor))
        # Resize the image
        zoomed_frame = cv2.resize(new_frame, new_size, interpolation = cv2.INTER_LINEAR)

        # Display the resulting frame
        show_in_moved_window('Input', zoomed_frame, 0, 10)
        show_in_moved_window('Input gray', new_frame_gray, 600, 10)
        show_in_moved_window('Difference image', dif_img, 1200, 10)
        show_in_moved_window('Binary_image', binary_image, 0, 400)

        # Old frame is updated
        frame_gray = alpha * frame_gray + (1 - alpha) * new_frame_gray

        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()
