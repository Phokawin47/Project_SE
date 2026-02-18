'use client'

import Image from "next/image";
import Header from '@/components/Header'
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <div className="min-h-screen">
      <Header />

        <div className="body">
          
          <img className="mx-auto mt-10" src="/Screenshot 2026-02-17 232456.png" alt="AI Illustration" />
          <br></br>
        
        </div>

      <Footer />
    </div>
  );
}
