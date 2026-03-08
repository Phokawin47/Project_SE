"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import WebcamCapture from "../../components/WebcamCapture";
import Link from "next/link";
import { ArrowLeftIcon } from "@heroicons/react/24/solid";
import NameEntryModal from "../../components/NameEntryModal";

export default function PlusChallenge() {
    const [gameState, setGameState] = useState<"READY" | "PLAYING" | "ENDED">("READY");
    const [showModal, setShowModal] = useState(false);
    const [score, setScore] = useState(0);
    const [timeLeft, setTimeLeft] = useState(60);
    const [targetSum, setTargetSum] = useState(0);

    const sfxRef = useRef<HTMLAudioElement | null>(null);

    // Tracking current detected fingers to prevent rapid re-triggering
    const [cooldown, setCooldown] = useState(false);

    const generateNewQuestion = useCallback(() => {
        const newSum = Math.floor(Math.random() * 11);
        setTargetSum(newSum);
    }, []);

    const startGame = () => {
        setScore(0);
        setTimeLeft(60);
        setGameState("PLAYING");
        generateNewQuestion();
        setCooldown(false);
    };

    useEffect(() => {
        let timer: NodeJS.Timeout;
        if (gameState === "PLAYING" && timeLeft > 0) {
            timer = setInterval(() => {
                setTimeLeft((prev) => prev - 1);
            }, 1000);
        } else if (timeLeft === 0 && gameState === "PLAYING") {
            setGameState("ENDED");
            setCooldown(true); // Stop accepting answers
        }
        return () => clearInterval(timer);
    }, [gameState, timeLeft]);

    const handleHandsDetected = useCallback((leftCount: number, rightCount: number) => {
        if (gameState !== "PLAYING" || cooldown) return;

        const totalFingers = leftCount + rightCount;

        if (totalFingers === targetSum) {
            // Lock out inputs temporarily
            setCooldown(true);

            // Correct answer logic
            setScore((prev) => prev + 1);

            // Play sound effect
            if (sfxRef.current) {
                sfxRef.current.currentTime = 0;
                sfxRef.current.play().catch((err: any) => console.warn("SFX play failed:", err));
            }

            // Wait 1.5s before showing next question to let user relax hands
            setTimeout(() => {
                generateNewQuestion();
                setCooldown(false);
            }, 1500);
        }
    }, [gameState, cooldown, targetSum, generateNewQuestion]);

    return (
        <div className="flex flex-col items-center w-full max-w-4xl mx-auto gap-8">
            {/* Sound Effect Audio Element */}
            <audio ref={sfxRef} src="/api/audio/get_point_sound.mp3" preload="auto" className="hidden" />

            <div className="w-full flex justify-between items-center bg-white p-4 rounded-2xl shadow-sm border">
                <Link href="/" className="text-slate-500 hover:text-indigo-600 flex items-center gap-2 font-medium transition-colors">
                    <ArrowLeftIcon className="w-5 h-5" /> Back to Home
                </Link>
                <h2 className="text-2xl font-bold text-slate-800">Funny with Plus 🧮</h2>
                <div className="w-24"></div> {/* Spacer for balancing flex */}
            </div>

            {gameState === "READY" && (
                <div className="flex flex-col items-center bg-white p-12 rounded-3xl shadow-xl border-4 border-indigo-100 text-center gap-6">
                    <h3 className="text-4xl font-black text-indigo-600">Ready to Add?</h3>
                    <p className="text-lg text-slate-600 font-medium">
                        1. A random number (0-10) will appear.<br />
                        2. Show that number of fingers using both hands.<br />
                        3. Answer as many as you can in 60 seconds!
                    </p>
                    <button
                        onClick={startGame}
                        className="mt-4 bg-indigo-600 hover:bg-indigo-700 text-white text-2xl font-bold py-4 px-12 rounded-full shadow-lg hover:shadow-xl transition-all hover:scale-105 active:scale-95"
                    >
                        Start Game!
                    </button>
                </div>
            )}

            {gameState === "PLAYING" && (
                <div className="w-full flex flex-col items-center gap-8">

                    <div className="flex w-full justify-between items-center text-2xl font-bold text-slate-700">
                        <div className="bg-white px-6 py-3 rounded-2xl shadow-md border-2 border-slate-100">
                            Time: <span className={`${timeLeft <= 10 ? 'text-red-500' : 'text-indigo-600'}`}>{timeLeft}s</span>
                        </div>
                        <div className="bg-white px-6 py-3 rounded-2xl shadow-md border-2 border-slate-100">
                            Score: <span className="text-green-600">{score}</span>
                        </div>
                    </div>

                    <div className="bg-gradient-to-br from-indigo-500 to-purple-600 p-12 rounded-[3rem] shadow-2xl w-full max-w-lg text-center transform transition-transform duration-300">
                        <p className="text-indigo-100 font-bold text-xl uppercase tracking-widest mb-4">Target Sum</p>
                        <h1 className="text-9xl font-black text-white drop-shadow-xl">{targetSum}</h1>
                    </div>

                    <div className="w-full opacity-100 transition-opacity duration-300">
                        {/* The Webcam view */}
                        <WebcamCapture onHandsDetected={handleHandsDetected} isActive={true} />
                    </div>

                </div>
            )}

            {gameState === "ENDED" && (
                <div className="flex flex-col items-center bg-white p-12 rounded-3xl shadow-xl border-4 border-green-100 text-center gap-6">
                    <h3 className="text-5xl font-black text-slate-800">Time's Up! ⏰</h3>
                    <div className="text-8xl font-black text-green-500 my-4 drop-shadow-md">{score}</div>
                    <p className="text-2xl text-slate-600 font-bold">Total Score</p>

                    <div className="flex gap-4 mt-8">
                        <button
                            onClick={startGame}
                            className="bg-green-500 hover:bg-green-600 text-white text-xl font-bold py-3 px-8 rounded-full shadow-md transition-all hover:scale-105"
                        >
                            Play Again
                        </button>
                        {/* Will link to leaderboard submission later */}
                        <button
                            onClick={() => setShowModal(true)}
                            className="bg-indigo-100 hover:bg-indigo-200 text-indigo-700 text-xl font-bold py-3 px-8 rounded-full shadow-md transition-all"
                        >
                            Save Score
                        </button>
                    </div>

                    {showModal && (
                        <NameEntryModal
                            score={score}
                            gameType="PLUS"
                            onClose={() => setShowModal(false)}
                        />
                    )}
                </div>
            )}

        </div>
    );
}
