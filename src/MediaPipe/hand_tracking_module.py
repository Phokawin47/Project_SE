import cv2
import mediapipe as mp
import math

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax

                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self, myHand):
        # myHand is the classification object (Left/Right)
        handType = myHand.classification[0].label
        fingers = []
        
        # 1. Detect Hand Orientation (Up/Down)
        # Compare Wrist (0) and Middle Finger MCP (9)
        # Y gets larger as we go down.
        # If Wrist.y > MiddleMCP.y -> Hand Pointing Up
        # If Wrist.y < MiddleMCP.y -> Hand Pointing Down
        
        y_wrist = self.lmList[0][2]
        y_middle_mcp = self.lmList[9][2]
        is_upright = y_wrist > y_middle_mcp
        
        # 2. Thumb Logic (Universal X-based)
        # Check orientation using Pinky MCP (17) vs Index MCP (5)
        x_pinky_mcp = self.lmList[17][1]
        x_index_mcp = self.lmList[5][1]
        
        # Determine expected thumb side
        # Upright Palm-in Right: Pinky(R) > Index(L) -> Thumb Left
        # Inverted Palm-in Right: Pinky(L) < Index(R) -> Thumb Right
        
        thumb_is_left = False
        if x_pinky_mcp > x_index_mcp:
             thumb_is_left = True
        
        # Check Thumb State
        # If Thumb is Left: Extended if Tip < IP
        # If Thumb is Right: Extended if Tip > IP
        if thumb_is_left:
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else: # Thumb is Right
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 3. 4 Fingers Logic (Dynamic Y-based)
        for id in range(1, 5):
            if is_upright:
                # Upright: Tip < Pip (Tip is higher)
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                # Inverted: Tip > Pip (Tip is lower)
                if self.lmList[self.tipIds[id]][2] > self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

    def get_hand_type(self, hand_index=0):
        if self.results.multi_handedness:
            if hand_index < len(self.results.multi_handedness):
                # "Right" or "Left"
                return self.results.multi_handedness[hand_index].classification[0].label
        return None
