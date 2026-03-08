import { FaceLandmarker, FilesetResolver, HandLandmarker, HandLandmarkerResult } from "@mediapipe/tasks-vision";

export class HandTrackingService {
  private handLandmarker: HandLandmarker | null = null;
  private runningMode: "IMAGE" | "VIDEO" = "VIDEO";

  constructor() {
    this.initializeHandLandmarker();
  }

  private async initializeHandLandmarker() {
    // Prevent execution during Next.js Server-Side Rendering
    if (typeof window === "undefined") {
      return;
    }

    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "/model/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: this.runningMode,
        numHands: 2,
        minHandDetectionConfidence: 0.75,
        minHandPresenceConfidence: 0.75,
        minTrackingConfidence: 0.75,
      });
      console.log("Hand landmarker initialized successfully");
    } catch (error) {
      console.error("Error initializing hand landmarker:", error);
    }
  }

  public async detectHands(videoElement: HTMLVideoElement, timestampMs: number): Promise<HandLandmarkerResult | null> {
    if (!this.handLandmarker) return null;
    return this.handLandmarker.detectForVideo(videoElement, timestampMs);
  }

  // Logic adapted from Python's fingersUp method
  public countFingers(result: HandLandmarkerResult) {
    let leftCount = 0;
    let rightCount = 0;

    if (!result.landmarks || result.landmarks.length === 0) {
      return { leftCount, rightCount };
    }

    const tipIds = [4, 8, 12, 16, 20];

    for (let i = 0; i < result.landmarks.length; i++) {
      const landmarks = result.landmarks[i];
      const handedness = result.handedness[i][0].categoryName; // "Left" or "Right"

      let fingers = [];

      // 1. Detect Hand Orientation
      const y_wrist = landmarks[0].y;
      const y_middle_mcp = landmarks[9].y;
      const is_upright = y_wrist > y_middle_mcp;

      // 2. Thumb Logic
      const x_pinky_mcp = landmarks[17].x;
      const x_index_mcp = landmarks[5].x;

      let thumb_is_left = x_pinky_mcp > x_index_mcp;

      if (thumb_is_left) {
        if (landmarks[tipIds[0]].x < landmarks[tipIds[0] - 1].x) {
          fingers.push(1);
        } else {
          fingers.push(0);
        }
      } else {
        if (landmarks[tipIds[0]].x > landmarks[tipIds[0] - 1].x) {
          fingers.push(1);
        } else {
          fingers.push(0);
        }
      }

      // 3. 4 Fingers Logic
      for (let id = 1; id < 5; id++) {
        if (is_upright) {
          if (landmarks[tipIds[id]].y < landmarks[tipIds[id] - 2].y) {
            fingers.push(1);
          } else {
            fingers.push(0);
          }
        } else {
          if (landmarks[tipIds[id]].y > landmarks[tipIds[id] - 2].y) {
            fingers.push(1);
          } else {
            fingers.push(0);
          }
        }
      }

      const count = fingers.filter(f => f === 1).length;

      if (handedness === "Left") {
        leftCount = count;
      } else if (handedness === "Right") {
        rightCount = count;
      }
    }

    return { leftCount, rightCount };
  }
}

// Export a singleton instance
export const handTrackingService = new HandTrackingService();
