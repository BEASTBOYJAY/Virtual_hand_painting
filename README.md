# Virtual Paint

Virtual Paint is an interactive application that enables users to create digital drawings using hand gestures. It leverages the MediaPipe library for hand tracking and OpenCV for video processing and visualization. The application supports different drawing colors and an eraser mode, allowing for a versatile virtual painting experience.

## Features

- Hand gesture recognition for drawing and color selection.
- Supports multiple colors and an eraser tool.
- Overlay images for color and tool selection.
- Real-time drawing on a virtual canvas.
- FPS (Frames Per Second) display for performance monitoring.
- New: now the brush size will change according to the hand distance 

## Components

### 1. `Hand_tracking_module.py`

This module defines the `HandDetector` class, which uses MediaPipe to detect hand landmarks and gestures in images. It provides functionality for detecting hands, finding landmark positions, and determining which fingers are up.

#### Key Methods:
- `FindHands(img, draw=True)`: Detects hands and optionally draws landmarks on the image.
- `FindPosition(img, handNo=0, draw=True)`: Finds and returns the positions of hand landmarks.
- `FingersUP()`: Determines which fingers are currently raised.

### 2. `virtual_paint.py`

This module implements the main application logic for the virtual painting tool. It uses the `HandDetector` class to process video frames from a webcam, enabling drawing and color selection through hand gestures.

#### Key Methods:
- `process_frame(img)`: Processes each frame for hand gestures and drawing actions.
- `combine_images(img)`: Combines the drawn canvas with the video frame.
- `add_fps(img)`: Adds the current FPS to the video frame.
- `run()`: Starts the virtual painter application and continuously processes video frames.

### 3. `requirements.txt`

This file lists the Python packages required to run the application. Ensure these packages are installed using pip.



## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/BEASTBOYJAY/Virtual_hand_painting.git
    cd Virtual_paint
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Place your overlay images in a folder named `header` within the project directory. These images will be used for color and tool selection.

## Usage

To start the virtual painter application, run:

```bash
python virtual_paint.py