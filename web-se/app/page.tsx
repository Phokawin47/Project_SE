import Link from "next/link";
import { CalculatorIcon, SpeakerWaveIcon } from "@heroicons/react/24/solid";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] gap-12 text-center">
      <div className="space-y-4">
        <h2 className="text-5xl font-black text-slate-800 tracking-tight leading-tight">
          Welcome to <span className="text-indigo-600">Funny Fingers</span>
        </h2>
        <p className="text-xl text-slate-600 max-w-2xl mx-auto font-medium">
          Learn math and listening skills using your hands! AI will detect your fingers in real-time.
        </p>
      </div>

      <div className="flex flex-col sm:flex-row gap-6 w-full max-w-3xl justify-center">
        {/* Game 1 Card */}
        <Link href="/games/plus" className="group flex-1 bg-white rounded-3xl p-8 shadow-xl border-2 border-transparent hover:border-indigo-500 hover:shadow-2xl hover:-translate-y-2 transition-all duration-300">
          <div className="bg-indigo-100 w-16 h-16 rounded-2xl flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform">
            <CalculatorIcon className="w-8 h-8 text-indigo-600" />
          </div>
          <h3 className="text-2xl font-bold text-slate-800 mb-2">Funny with Plus</h3>
          <p className="text-slate-500 font-medium">Solve the math problem by showing the correct number of fingers (0-10).</p>
        </Link>

        {/* Game 2 Card */}
        <Link href="/games/listen" className="group flex-1 bg-white rounded-3xl p-8 shadow-xl border-2 border-transparent hover:border-pink-500 hover:shadow-2xl hover:-translate-y-2 transition-all duration-300">
          <div className="bg-pink-100 w-16 h-16 rounded-2xl flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform">
            <SpeakerWaveIcon className="w-8 h-8 text-pink-600" />
          </div>
          <h3 className="text-2xl font-bold text-slate-800 mb-2">Funny with Listening</h3>
          <p className="text-slate-500 font-medium">Listen to the number and show it with your hands (Tens on left, Ones on right).</p>
        </Link>
      </div>
    </div>
  );
}
