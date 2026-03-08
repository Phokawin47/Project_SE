"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import WebcamCapture from "../../components/WebcamCapture";
import Link from "next/link";
import { ArrowLeftIcon, SpeakerWaveIcon } from "@heroicons/react/24/solid";
import NameEntryModal from "../../components/NameEntryModal";

export default function ListeningChallenge() {
    const [gameState, setGameState] = useState<"READY" | "PLAYING" | "ENDED">("READY");
    const [showModal, setShowModal] = useState(false);
    const [score, setScore] = useState(0);
    const [timeLeft, setTimeLeft] = useState(60);
    const [targetNumber, setTargetNumber] = useState(0);
    const [targetTens, setTargetTens] = useState(0);
    const [targetOnes, setTargetOnes] = useState(0);

    const [cooldown, setCooldown] = useState(false);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const sfxRef = useRef<HTMLAudioElement | null>(null);

    const generateNewQuestion = useCallback(() => {
        // Generate numbers 0-55 where tens <= 5 and ones <= 5
        const tens = Math.floor(Math.random() * 6);
        const ones = Math.floor(Math.random() * 6);

        // Combining tens and ones
        const newNumber = (tens * 10) + ones;

        setTargetTens(tens);
        setTargetOnes(ones);
        setTargetNumber(newNumber);

        // Play audio
        playNumberAudio(newNumber);

    }, []);

    const playNumberAudio = (number: number) => {
        // Fetch audio from our MongoDB GridFS API
        const audioPath = `/api/audio/${number}.mp3`;

        if (audioRef.current) {
            // Stop current playback before playing new one to prevent overlapping
            audioRef.current.pause();
            audioRef.current.currentTime = 0;

            audioRef.current.src = audioPath;
            audioRef.current.play().catch((err) => {
                console.warn("Failed to play audio from database:", err);
            });
        }
    };

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

        // Based on user requirements: Tens = Left Hand, Ones = Right Hand
        if (leftCount === targetTens && rightCount === targetOnes) {
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
    }, [gameState, cooldown, targetTens, targetOnes, generateNewQuestion]);

    return (
        <div className="flex flex-col items-center w-full max-w-4xl mx-auto gap-8">
            {/* Hidden Audio Elements */}
            <audio ref={audioRef} className="hidden" />
            <audio ref={sfxRef} src="/api/audio/get_point_sound.mp3" preload="auto" className="hidden" />

            <div className="w-full flex justify-between items-center bg-white p-4 rounded-2xl shadow-sm border">
                <Link href="/" className="text-slate-500 hover:text-pink-600 flex items-center gap-2 font-medium transition-colors">
                    <ArrowLeftIcon className="w-5 h-5" /> Back to Home
                </Link>
                <h2 className="text-2xl font-bold text-slate-800">Funny with Listening 🎧</h2>
                <div className="w-24"></div> {/* Spacer */}
            </div>

            {gameState === "READY" && (
                <div className="flex flex-col items-center bg-white p-12 rounded-3xl shadow-xl border-4 border-pink-100 text-center gap-6">
                    <h3 className="text-4xl font-black text-pink-600">Ready to Listen?</h3>
                    <p className="text-lg text-slate-600 font-medium max-w-lg">
                        1. You will hear a number between 0 and 55.<br />
                        2. Show the Tens digit using your Left hand.<br />
                        3. Show the Ones digit using your Right hand.<br />
                        (Example: For "34", show 3 fingers on the left and 4 on the right)<br />
                        4. 60 Seconds on the clock!
                    </p>
                    <button
                        onClick={startGame}
                        className="mt-4 bg-pink-600 hover:bg-pink-700 text-white text-2xl font-bold py-4 px-12 rounded-full shadow-lg hover:shadow-xl transition-all hover:scale-105 active:scale-95"
                    >
                        Start Challenge!
                    </button>
                </div>
            )}

            {gameState === "PLAYING" && (
                <div className="w-full flex flex-col items-center gap-8">

                    <div className="flex w-full justify-between items-center text-2xl font-bold text-slate-700">
                        <div className="bg-white px-6 py-3 rounded-2xl shadow-md border-2 border-slate-100">
                            Time: <span className={`${timeLeft <= 10 ? 'text-red-500' : 'text-pink-600'}`}>{timeLeft}s</span>
                        </div>

                        {/* Center play button to repeat sound */}
                        <button
                            onClick={() => playNumberAudio(targetNumber)}
                            className="bg-pink-100 p-3 rounded-full hover:bg-pink-200 text-pink-600 shadow-sm border border-pink-200 transition-colors"
                            title="Hear Again"
                        >
                            <SpeakerWaveIcon className="w-8 h-8" />
                        </button>

                        <div className="bg-white px-6 py-3 rounded-2xl shadow-md border-2 border-slate-100">
                            Score: <span className="text-green-600">{score}</span>
                        </div>
                    </div>

                    {/* Visualization to help player know it's recording */}
                    <div className="bg-gradient-to-br from-pink-400 to-rose-500 p-8 rounded-[3rem] shadow-xl w-full max-w-lg text-center transform transition-transform duration-300">
                        <div className="flex justify-center mb-4">
                            <SpeakerWaveIcon className="w-16 h-16 text-white animate-pulse opacity-80" />
                        </div>
                        <p className="text-pink-100 font-bold text-xl uppercase tracking-widest">Listen Carefully...</p>
                        {/* Note: Target number is NOT displayed visually to enforce listening */}
                    </div>

                    <div className="w-full opacity-100 transition-opacity duration-300">
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
                        <button
                            onClick={() => setShowModal(true)}
                            className="bg-pink-100 hover:bg-pink-200 text-pink-700 text-xl font-bold py-3 px-8 rounded-full shadow-md transition-all"
                        >
                            Save Score
                        </button>
                    </div>

                    {showModal && (
                        <NameEntryModal
                            score={score}
                            gameType="LISTEN"
                            onClose={() => setShowModal(false)}
                        />
                    )}
                </div>
            )}

        </div>
    );
}
