"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeftIcon, TrophyIcon } from "@heroicons/react/24/solid";

interface ScoreData {
    _id: string;
    nickname: string;
    score: number;
    gameType: "PLUS" | "LISTEN";
    createdAt: string;
}

export default function Leaderboard() {
    const [activeTab, setActiveTab] = useState<"PLUS" | "LISTEN">("PLUS");
    const [scores, setScores] = useState<ScoreData[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchScores = async () => {
            setLoading(true);
            try {
                const res = await fetch(`/api/scores?gameType=${activeTab}`);
                const json = await res.json();
                if (json.success) {
                    setScores(json.data);
                }
            } catch (error) {
                console.error("Failed to fetch scores", error);
            } finally {
                setLoading(false);
            }
        };

        fetchScores();
    }, [activeTab]);

    return (
        <div className="flex flex-col items-center w-full max-w-3xl mx-auto gap-8 py-8">

            <div className="w-full flex justify-between items-center bg-white p-4 rounded-2xl shadow-sm border">
                <Link href="/" className="text-slate-500 hover:text-indigo-600 flex items-center gap-2 font-medium transition-colors">
                    <ArrowLeftIcon className="w-5 h-5" /> Back to Home
                </Link>
                <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
                    <TrophyIcon className="w-6 h-6 text-yellow-500" /> Hall of Fame
                </h2>
                <div className="w-24"></div>
            </div>

            <div className="bg-white w-full rounded-3xl shadow-xl border overflow-hidden">
                {/* Tabs */}
                <div className="flex border-b">
                    <button
                        onClick={() => setActiveTab("PLUS")}
                        className={`flex-1 py-4 text-lg font-bold transition-colors ${activeTab === "PLUS" ? "text-indigo-600 border-b-4 border-indigo-600 bg-indigo-50/50" : "text-slate-400 hover:text-slate-600 hover:bg-slate-50"}`}
                    >
                        Funny with Plus
                    </button>
                    <button
                        onClick={() => setActiveTab("LISTEN")}
                        className={`flex-1 py-4 text-lg font-bold transition-colors ${activeTab === "LISTEN" ? "text-pink-600 border-b-4 border-pink-600 bg-pink-50/50" : "text-slate-400 hover:text-slate-600 hover:bg-slate-50"}`}
                    >
                        Funny with Listening
                    </button>
                </div>

                {/* List */}
                <div className="px-6 pb-6 pt-4 min-h-[400px]">
                    {loading ? (
                        <div className="flex justify-center items-center h-full pt-20">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
                        </div>
                    ) : scores.length === 0 ? (
                        <div className="text-center text-slate-400 mt-20 font-medium">
                            No scores yet. Be the first to play!
                        </div>
                    ) : (
                        <div className="space-y-3 mt-4">
                            {scores.map((score, index) => (
                                <div key={score._id} className="flex items-center bg-slate-50 border rounded-2xl p-4 transition-transform hover:-translate-y-1 hover:shadow-md">
                                    <div className={`w-10 h-10 rounded-full flex items-center justify-center font-black text-lg mr-4 ${index === 0 ? "bg-yellow-100 text-yellow-600" :
                                            index === 1 ? "bg-slate-200 text-slate-600" :
                                                index === 2 ? "bg-orange-100 text-orange-600" : "bg-white text-slate-400 border"
                                        }`}>
                                        {index + 1}
                                    </div>
                                    <div className="flex-1 font-bold text-slate-800 text-xl">{score.nickname}</div>
                                    <div className="text-3xl font-black text-indigo-600">{score.score}</div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

        </div>
    );
}
