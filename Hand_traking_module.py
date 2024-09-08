import cv2
import mediapipe as mp


class HandDetector:
    """
    A class to detect hands and retrieve hand landmarks in an image using MediaPipe.

    Attributes:
        mode (bool): Whether to treat the input images as a batch of static images.
        maxHands (int): The maximum number of hands to detect.
        detection_confidence (float): Minimum confidence value for hand detection.
        tracking_confidence (float): Minimum confidence value for hand tracking.
        mpHands: MediaPipe hands solution.
        hands: MediaPipe Hands object for processing images.
        mpDraw: MediaPipe utility for drawing landmarks on the image.
        tipIDs (list): List of landmark IDs corresponding to finger tips.
    """

    def __init__(
        self, mode=False, maxHands=2, detection_confidence=0.5, tracking_confidence=0.5
    ):
        """
        Initializes the HandDetector class.

        Args:
            mode (bool, optional): If True, treats the input images as a static image batch. Defaults to False.
            maxHands (int, optional): Maximum number of hands to detect. Defaults to 2.
            detection_confidence (float, optional): Minimum confidence for hand detection. Defaults to 0.5.
            tracking_confidence (float, optional): Minimum confidence for hand tracking. Defaults to 0.5.
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize Mediapipe hands solution
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIDs = [4, 8, 12, 16, 20]

    def FindHands(self, img, draw=True):
        """
        Detects hands in the input image and optionally draws landmarks.

        Args:
            img (ndarray): The image in which hands are to be detected.
            draw (bool, optional): If True, draws hand landmarks on the image. Defaults to True.

        Returns:
            ndarray: The processed image with drawn hand landmarks (if draw=True).
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLMS, self.mpHands.HAND_CONNECTIONS
                    )

        return img

    def FindPosition(self, img, handNo=0, draw=True):
        """
        Finds the position of hand landmarks in the input image.

        Args:
            img (ndarray): The image in which the hand landmarks are located.
            handNo (int, optional): Index of the hand to analyze. Defaults to 0.
            draw (bool, optional): If True, draws circles at landmark positions. Defaults to True.

        Returns:
            list: A list of landmark positions in the format [id, cx, cy], where `id` is the landmark index,
                  and `cx`, `cy` are the x and y coordinates in pixels.
        """
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                # Get the position in pixels
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return self.lmlist

    def FingersUP(self):
        """
        Determines which fingers are up based on landmark positions.

        Returns:
            list: A list of integers representing the state of each finger (1 for up, 0 for down).
        """
        fingers = []

        # Thumb
        if self.lmlist[self.tipIDs[0]][1] < self.lmlist[self.tipIDs[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # For 4 fingers
        for id in range(1, 5):
            if self.lmlist[self.tipIDs[id]][2] < self.lmlist[self.tipIDs[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
