import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Funny Fingers - Math & Listening Games",
  description: "An educational web application using AI to track fingers for math and listening games.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`antialiased bg-slate-50 text-slate-900 min-h-screen flex flex-col font-sans`} suppressHydrationWarning>
        <header className="bg-white shadow-sm border-b sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
            <h1 className="text-2xl font-black text-indigo-600 tracking-tight flex items-center gap-2">
              <span className="text-3xl">✌️</span> Funny Fingers
            </h1>
            <nav className="flex gap-4">
              <a href="/" className="font-semibold text-slate-600 hover:text-indigo-600 transition-colors">Home</a>
              <a href="/leaderboard" className="font-semibold text-slate-600 hover:text-indigo-600 transition-colors">Leaderboard</a>
            </nav>
          </div>
        </header>

        <main className="flex-1 w-full max-w-7xl mx-auto p-4 sm:p-6 lg:p-8 flex flex-col">
          {children}
        </main>

        <footer className="bg-slate-100 py-6 text-center text-sm text-slate-500 font-medium">
          <p>© 2026 Funny Fingers Educational Play</p>
        </footer>
      </body>
    </html>
  );
}
