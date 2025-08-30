import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Canvas and brush settings
canvas = None
brushThickness = 10
eraserThickness = 50
drawColor = (255, 0, 0)  # default color

# Color palette
colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255)]
colorIndex = 0

def draw_palette(img):
    x = 50
    for i, color in enumerate(colors):
        cv2.rectangle(img, (x,10), (x+50,60), color, -1)
        if i == colorIndex:
            cv2.rectangle(img, (x,10), (x+50,60), (255,255,255), 3)  # selected border
        x += 70

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)

    # Draw color palette
    draw_palette(img)

    # Detect hands
    hands, img = detector.findHands(img)
    if hands:
        lmList = hands[0]['lmList']
        fingers = detector.fingersUp(hands[0])
        x1, y1 = lmList[8][0], lmList[8][1]  # Index tip
        x2, y2 = lmList[4][0], lmList[4][1]  # Thumb tip

        # Check color selection
        if y1 < 60:  # Top area for palette
            x = 50
            for i in range(len(colors)):
                if x < x1 < x+50:
                    colorIndex = i
                    drawColor = colors[colorIndex]
                x += 70

        # Clear canvas gesture (index finger top-left corner)
        if x1 < 50 and y1 < 50:
            canvas = np.zeros_like(img)

        # Eraser mode (fist: all fingers down)
        if fingers == [0,0,0,0,0]:
            cv2.circle(canvas, (x1, y1), eraserThickness, (0,0,0), -1)
        # Drawing mode (index finger up, middle finger down)
        elif fingers[0]==1 and fingers[1]==0:
            cv2.circle(canvas, (x1, y1), brushThickness, drawColor, -1)

    # Merge canvas with camera feed
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Virtual Drawing Board", img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('s'):  # Press 's' to save drawing
        cv2.imwrite("my_drawing.png", canvas)
        print("Drawing saved as my_drawing.png")

cap.release()
cv2.destroyAllWindows()
