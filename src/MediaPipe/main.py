import cv2
import time
import os
import hand_tracking_module as htm

def main():
    import random
    import pyttsx3
    import threading

    # Initialize TTS
    engine = pyttsx3.init()
    
    def speak(text):
        def _speak():
            try:
                engine_thread = pyttsx3.init() # Re-init in thread or use lock? pyttsx3 is not thread safe usually. 
                # Better: Keep one engine in main, use queue? Or just re-init for simple tasks.
                # Actually, pyttsx3 on Windows (SAPI5) often works if initialized in thread.
                # Let's try simple runAndWait in thread.
                engine_thread.say(text)
                engine_thread.runAndWait()
            except:
                pass
        threading.Thread(target=_speak).start()

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280) # Width
    cap.set(4, 720)  # Height
    detector = htm.HandDetector(detectionCon=0.75, maxHands=2)
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        # Flip image for mirror view
        img = cv2.flip(img, 1)
        
        img = detector.findHands(img)
        
        if detector.results.multi_hand_landmarks:
            for handNo, hand_landmarks in enumerate(detector.results.multi_hand_landmarks):
                # Get landmarks and bounding box
                lmList, bbox = detector.findPosition(img, handNo=handNo, draw=True)
                
                if detector.results.multi_handedness and handNo < len(detector.results.multi_handedness):
                    # Hand Classification
                    # Note: Since we flipped the image, MediaPipe sees the "mirror" version.
                    # A real 'Right' hand appears as a 'Left' hand in the flipped image.
                    # So we swap the label for display to match user's perspective (Mirror).
                    
                    mp_label = detector.results.multi_handedness[handNo].classification[0].label
                    display_label = "Right" if mp_label == "Left" else "Left"
                    
                    # Confidence score
                    score = detector.results.multi_handedness[handNo].classification[0].score
                    
                    if len(lmList) != 0:
                        # Count fingers
                        # We pass the original MediaPipe label because fingersUp logic might depend on geometry of that 'perceived' hand
                        fingers = detector.fingersUp(detector.results.multi_handedness[handNo])
                        count = fingers.count(1)
                        
                        # Draw Info
                        cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
                        
                        # Text Background
                        cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 80), (bbox[2] + 20, bbox[1] - 20), (0, 255, 0), cv2.FILLED)
                        
                        info_text = f"{display_label}: {count}"
                        cv2.putText(img, info_text, (bbox[0] - 15, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                        
                        # Debug info for thumb tip vs IP
                        # thumb_tip_x = lmList[4][1]
                        # thumb_ip_x = lmList[3][1]
                        # cv2.putText(img, f"T:{thumb_tip_x} IP:{thumb_ip_x}", (bbox[0], bbox[3] + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # Draw Title
        cv2.putText(img, "Hand Detection Mode", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
