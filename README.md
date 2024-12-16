# Air-Canvas
# Hand-Gesture-Based Drawing and Recognition App

This project is a hand-gesture-based drawing application that lets users draw on a virtual canvas using their fingers. It also incorporates OCR (Optical Character Recognition) to recognize text from the drawn content using Tesseract.

## Features

- **Hand-Gesture Drawing**: Draw on a virtual canvas using finger gestures captured via webcam.
- **Color Selection**: Choose from different colors (blue, green, red, yellow) using virtual buttons.
- **Clear Canvas**: Erase all content on the canvas with a single gesture.
- **Text Recognition**: Recognize text drawn on the canvas using Tesseract OCR.

## Requirements

Make sure you have the following libraries installed:

- `opencv-python`
- `numpy`
- `mediapipe`
- `pytesseract`

To install the required libraries, run:

```bash
pip install opencv-python numpy mediapipe pytesseract
```

Additionally, you need Tesseract OCR installed on your system. You can download it from [Tesseract's official page](https://github.com/tesseract-ocr/tesseract). Once installed, set the `pytesseract.pytesseract.tesseract_cmd` variable to the path of the Tesseract executable (e.g., `C:/Program Files/Tesseract-OCR/tesseract.exe` on Windows).

## How It Works

1. **Hand Tracking**: The app uses MediaPipe's Hands module to track hand landmarks.
2. **Gesture Recognition**: The forefinger tip is used to draw on the canvas, and the thumb is used for gesture-based actions(i.e. pinch sign is used to stop drawing on the canvas).
3. **Color and Action Selection**: Virtual buttons allow users to change the drawing color or clear the canvas.
4. **Text Recognition**: Press the 'R' key to recognize text drawn on the canvas using Tesseract OCR.

## Usage

1. Run the script:
   ```bash
   python gesture_drawing_app.py
   ```
2. Use your index finger to draw on the canvas.
3. Hover your finger over the virtual buttons to:
   - Clear the canvas ("CLEAR" button).
   - Change the drawing color to blue, green, red, or yellow.
4. Press the 'R' key to recognize any text drawn on the canvas.
5. Press the 'Q' key to quit the application.


## Limitations

- Hand tracking accuracy depends on lighting and camera quality.
- Tesseract OCR may not always recognize handwritten text accurately.

## Future Improvements

- Enhance OCR accuracy for handwritten text.
- Add more gesture-based functionalities, such as saving drawings.
- Improve tracking under low-light conditions.

## Author
Parth Lathiya
