import { NextResponse } from 'next/server';
import dbConnect from '../../../lib/mongodb';
import Score from '../../../lib/Score';

// GET top scores
export async function GET(request: Request) {
    try {
        const { searchParams } = new URL(request.url);
        const gameType = searchParams.get('gameType'); // "PLUS" or "LISTEN"

        await dbConnect();

        // Query filters
        const filter = gameType ? { gameType } : {};

        // Get top 10 scores
        const scores = await Score.find(filter)
            .sort({ score: -1, createdAt: 1 }) // Highest score first, then earliest time
            .limit(10);

        return NextResponse.json({ success: true, data: scores });
    } catch (error: any) {
        return NextResponse.json({ success: false, error: error.message }, { status: 400 });
    }
}

// POST a new score
export async function POST(request: Request) {
    try {
        const body = await request.json();
        await dbConnect();

        const score = await Score.create(body);
        return NextResponse.json({ success: true, data: score }, { status: 201 });
    } catch (error: any) {
        return NextResponse.json({ success: false, error: error.message }, { status: 400 });
    }
}
