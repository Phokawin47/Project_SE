"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";

interface NameEntryModalProps {
    score: number;
    gameType: "PLUS" | "LISTEN";
    onClose: () => void;
}

export default function NameEntryModal({ score, gameType, onClose }: NameEntryModalProps) {
    const [nickname, setNickname] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);
    const router = useRouter();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!nickname.trim()) return;

        setIsSubmitting(true);

        try {
            const res = await fetch("/api/scores", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    nickname: nickname.trim(),
                    score,
                    gameType,
                }),
            });

            if (res.ok) {
                // Redirect to leaderboard
                router.push("/leaderboard");
            } else {
                console.error("Failed to save score");
                setIsSubmitting(false);
            }
        } catch (error) {
            console.error(error);
            setIsSubmitting(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-3xl p-8 shadow-2xl w-full max-w-md transform scale-100 animate-in zoom-in-95 duration-200">
                <h2 className="text-3xl font-black text-indigo-600 mb-2 text-center">New High Score!</h2>
                <p className="text-slate-500 text-center mb-6 font-medium">Record your victory on the leaderboard.</p>

                <div className="bg-indigo-50 border-2 border-indigo-100 rounded-2xl p-6 mb-8 text-center">
                    <div className="text-sm text-indigo-400 font-bold uppercase tracking-wider mb-1">Your Score</div>
                    <div className="text-6xl font-black text-indigo-600 drop-shadow-sm">{score}</div>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label htmlFor="nickname" className="block text-sm font-bold text-slate-700 mb-2">Enter Nickname (max 10 chars)</label>
                        <input
                            id="nickname"
                            type="text"
                            maxLength={10}
                            required
                            value={nickname}
                            onChange={(e) => setNickname(e.target.value)}
                            className="w-full text-center text-2xl font-bold text-slate-800 bg-slate-50 border-2 border-slate-200 rounded-xl px-4 py-4 focus:outline-none focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100 transition-all placeholder:text-slate-300 placeholder:font-medium"
                            placeholder="Player 1"
                        />
                    </div>

                    <div className="flex gap-3 pt-4">
                        <button
                            type="button"
                            onClick={onClose}
                            disabled={isSubmitting}
                            className="flex-1 bg-slate-100 hover:bg-slate-200 text-slate-600 font-bold py-3 rounded-xl transition-colors disabled:opacity-50"
                        >
                            Skip
                        </button>
                        <button
                            type="submit"
                            disabled={isSubmitting || !nickname.trim()}
                            className="flex-[2] bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 rounded-xl shadow-md transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isSubmitting ? "Saving..." : "Save Score"}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}
