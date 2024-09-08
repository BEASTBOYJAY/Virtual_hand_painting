import cv2
import os
import numpy as np
import time
from Hand_traking_module import HandDetector


class VirtualPainter:
    """
    A class to implement a virtual painting application using hand gestures.

    Attributes:
        folder_path (str): Path to the folder containing overlay images.
        overlay_lst (list): List of overlay images used for UI.
        header (numpy.ndarray): Current header image.
        drawing_color (tuple): Current drawing color in BGR format.
        brush_size (int): Size of the drawing brush.
        eraser_size (int): Size of the eraser brush.
        img_canvas (numpy.ndarray): Canvas for drawing.
        xp (int): Previous x-coordinate of the drawing point.
        yp (int): Previous y-coordinate of the drawing point.
        cap (cv2.VideoCapture): Video capture object for webcam.
        detector (HandDetector): Hand detector instance.
        prev_time (float): Previous frame timestamp for FPS calculation.
        current_time (float): Current frame timestamp for FPS calculation.
    """

    def __init__(self, folder_path="header"):
        """
        Initializes the VirtualPainter with the given folder path for overlay images.

        Args:
            folder_path (str): Path to the folder containing overlay images. Defaults to "header".
        """
        # Initialize parameters
        self.folder_path = folder_path
        self.overlay_lst = self.load_overlays()
        self.header = self.overlay_lst[0]
        self.drawing_color = (0, 0, 255)  # Red
        self.brush_size = 8
        self.eraser_size = 50
        self.img_canvas = np.zeros((720, 1280, 3), np.uint8)
        self.xp, self.yp = 0, 0

        # Set up video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)  # Width
        self.cap.set(4, 720)  # Height

        # Initialize Hand Detector
        self.detector = HandDetector(detection_confidence=0.85)

        # Initialize FPS variables
        self.prev_time = 0
        self.current_time = 0

    def load_overlays(self):
        """
        Loads overlay images from the specified folder.

        Returns:
            list: A list of images loaded as overlays.
        """
        overlay_lst = []
        header_lst = os.listdir(self.folder_path)
        for img_path in header_lst:
            header_img = cv2.imread(os.path.join(self.folder_path, img_path))
            overlay_lst.append(header_img)
        return overlay_lst

    def process_frame(self, img):
        """
        Processes a single video frame, detecting hands and drawing on the canvas.

        Args:
            img (numpy.ndarray): The input video frame.

        Returns:
            numpy.ndarray: The processed video frame with drawings.
        """
        img = cv2.flip(img, 1)
        img = self.detector.FindHands(img)
        lmlist = self.detector.FindPosition(img, draw=False)

        if len(lmlist) != 0:
            # tip of index and middle fingers
            x1, y1 = lmlist[8][1:]
            x2, y2 = lmlist[12][1:]

            # Check which fingers are up
            fingers = self.detector.FingersUP()

            # If Selection mode - 2 fingers are up
            if fingers[1] and fingers[2]:
                self.xp, self.yp = 0, 0
                if y1 < 110:
                    if 50 < x1 < 250:
                        self.header = self.overlay_lst[0]
                        self.drawing_color = (0, 0, 255)  # Red
                    elif 375 < x1 < 575:
                        self.header = self.overlay_lst[1]
                        self.drawing_color = (0, 255, 0)  # Green
                    elif 700 < x1 < 900:
                        self.header = self.overlay_lst[3]
                        self.drawing_color = (255, 0, 255)  # Pink
                    elif 1025 < x1 < 1175:
                        self.header = self.overlay_lst[2]
                        self.drawing_color = (0, 0, 0)  # Eraser
                cv2.rectangle(
                    img, (x1, y1 - 25), (x2, y2 + 25), self.drawing_color, cv2.FILLED
                )

            # If Drawing mode - 1 finger is up
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, self.drawing_color, cv2.FILLED)

                if self.xp == 0 and self.yp == 0:
                    self.xp, self.yp = x1, y1

                if self.drawing_color == (0, 0, 0):
                    cv2.line(
                        img,
                        (self.xp, self.yp),
                        (x1, y1),
                        self.drawing_color,
                        self.eraser_size,
                    )
                    cv2.line(
                        self.img_canvas,
                        (self.xp, self.yp),
                        (x1, y1),
                        self.drawing_color,
                        self.eraser_size,
                    )
                else:
                    cv2.line(
                        img,
                        (self.xp, self.yp),
                        (x1, y1),
                        self.drawing_color,
                        self.brush_size,
                    )
                    cv2.line(
                        self.img_canvas,
                        (self.xp, self.yp),
                        (x1, y1),
                        self.drawing_color,
                        self.brush_size,
                    )

                self.xp, self.yp = x1, y1

        return img

    def combine_images(self, img):
        """
        Combines the drawn canvas with the current video frame.

        Args:
            img (numpy.ndarray): The input video frame.

        Returns:
            numpy.ndarray: The combined frame with the drawings and overlay.
        """
        img_gray = cv2.cvtColor(self.img_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inverse = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inverse = cv2.cvtColor(img_inverse, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inverse)
        img = cv2.bitwise_or(img, self.img_canvas)
        img[0:100, 0:1280] = self.header
        return img

    def add_fps(self, img):
        """
        Adds the current FPS (frames per second) to the video frame.

        Args:
            img (numpy.ndarray): The input video frame.

        Returns:
            numpy.ndarray: The video frame with the FPS added.
        """
        self.current_time = time.time()
        fps = 1 / (self.current_time - self.prev_time)
        self.prev_time = self.current_time

        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        return img

    def run(self):
        """
        Runs the virtual painter application, continuously processing video frames.
        """
        while True:
            success, img = self.cap.read()
            if not success:
                break

            img = self.process_frame(img)
            img = self.combine_images(img)
            img = self.add_fps(img)

            # Show the image
            cv2.imshow("Virtual Painter", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    painter = VirtualPainter()
    painter.run()
