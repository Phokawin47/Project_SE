"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { HandTrackingService } from "../services/handTracking";

interface WebcamCaptureProps {
    onHandsDetected: (leftCount: number, rightCount: number) => void;
    isActive: boolean;
}

export default function WebcamCapture({ onHandsDetected, isActive }: WebcamCaptureProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const requestRef = useRef<number>(null);
    const statsRef = useRef<HTMLDivElement>(null);
    const onHandsDetectedRef = useRef(onHandsDetected);
    const [trackerService, setTrackerService] = useState<HandTrackingService | null>(null);
    const [deviceError, setDeviceError] = useState<string | null>(null);

    // Initialize service
    useEffect(() => {
        const service = new HandTrackingService();
        setTrackerService(service);
    }, []);

    // Update the ref to point to the newest callback
    useEffect(() => {
        onHandsDetectedRef.current = onHandsDetected;
    }, [onHandsDetected]);

    // Setup Webcam
    useEffect(() => {
        const startCamera = async () => {
            if (!isActive) return;
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        facingMode: "user" // front camera
                    }
                });
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            } catch (err: any) {
                console.error("Camera access denied or unavailable", err);
                setDeviceError("Camera access denied or not found. Please check permissions.");
            }
        };

        if (isActive) {
            startCamera();
        } else {
            // Stop camera if not active
            if (videoRef.current && videoRef.current.srcObject) {
                const stream = videoRef.current.srcObject as MediaStream;
                stream.getTracks().forEach(track => track.stop());
                videoRef.current.srcObject = null;
            }
            if (requestRef.current) {
                cancelAnimationFrame(requestRef.current);
            }
        }
    }, [isActive]);

    const predictWebcam = () => {
        if (!trackerService || !videoRef.current || !isActive) return;

        // Check if video is playing
        if (videoRef.current.currentTime > 0 && !videoRef.current.paused && !videoRef.current.ended) {
            const startTimeMs = performance.now();

            trackerService.detectHands(videoRef.current, startTimeMs).then((results) => {
                if (results) {
                    // Count fingers
                    const { leftCount, rightCount } = trackerService.countFingers(results);
                    onHandsDetectedRef.current(leftCount, rightCount);

                    if (statsRef.current) {
                        statsRef.current.innerText = `L: ${leftCount} | R: ${rightCount}`;
                    }

                    // Draw landmarks 
                    const canvasCtx = canvasRef.current?.getContext("2d");
                    if (canvasCtx && canvasRef.current && videoRef.current) {
                        canvasRef.current.width = videoRef.current.videoWidth;
                        canvasRef.current.height = videoRef.current.videoHeight;

                        canvasCtx.save();
                        canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

                        // Mirror the canvas context
                        canvasCtx.translate(canvasRef.current.width, 0);
                        canvasCtx.scale(-1, 1);

                        if (results.landmarks) {
                            for (const landmarks of results.landmarks) {
                                // Simple drawing for visualization, better to use MediaPipe Drawing Utils if needed
                                canvasCtx.fillStyle = "#FF0000";
                                for (const landmark of landmarks) {
                                    canvasCtx.beginPath();
                                    canvasCtx.arc(landmark.x * canvasRef.current.width, landmark.y * canvasRef.current.height, 5, 0, 2 * Math.PI);
                                    canvasCtx.fill();
                                }
                            }
                        }
                        canvasCtx.restore();
                    }
                }

                // Call next frame only AFTER current processing is done to prevent queue buildup
                if (isActive) {
                    requestRef.current = requestAnimationFrame(predictWebcam);
                }
            }).catch(err => {
                console.error("Hand tracking error:", err);
                if (isActive) {
                    requestRef.current = requestAnimationFrame(predictWebcam);
                }
            });
        } else {
            // If video isn't ready yet, just try again next frame
            if (isActive) {
                requestRef.current = requestAnimationFrame(predictWebcam);
            }
        }
    };

    useEffect(() => {
        if (isActive && trackerService && videoRef.current) {
            // Start the loop
            requestRef.current = requestAnimationFrame(predictWebcam);
        }
        return () => {
            if (requestRef.current) {
                cancelAnimationFrame(requestRef.current);
            }
        }
    }, [isActive, trackerService]); // removed predictWebcam dependency since we define it inside or handle it differently

    if (deviceError) {
        return (
            <div className="flex items-center justify-center p-6 bg-red-100 rounded-xl text-red-700 font-medium">
                {deviceError}
            </div>
        );
    }

    return (
        <div className="relative w-full max-w-4xl mx-auto rounded-3xl overflow-hidden shadow-2xl border-4 border-white/20">
            {/* Video element - visually hidden but used for prediction. We show the canvas instead. */}
            <video
                ref={videoRef}
                autoPlay
                playsInline
                className="w-full h-auto transform scale-x-[-1]"
                style={{ display: 'block', position: 'absolute', top: 0, left: 0, opacity: 0.5, zIndex: 0 }}
            />
            {/* Canvas element for overlaid drawings */}
            <canvas
                ref={canvasRef}
                className="w-full h-auto z-10 relative pointer-events-none"
                style={{ display: 'block' }}
            />

            <div className="absolute top-4 left-4 bg-black/50 backdrop-blur-md px-4 py-2 rounded-full text-white text-sm font-medium z-20 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
                AI Camera Active
            </div>

            {/* Real-time Hands Stat Overlay */}
            <div
                ref={statsRef}
                className="absolute top-4 right-4 bg-indigo-600/90 backdrop-blur-md px-5 py-2 rounded-2xl text-white text-xl font-bold tracking-wider z-20 shadow-xl border-2 border-indigo-400/50"
            >
                L: 0 | R: 0
            </div>
        </div>
    );
}
