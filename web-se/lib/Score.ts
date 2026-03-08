import mongoose, { Schema, Document } from 'mongoose';

export interface IScore extends Document {
    nickname: string;
    gameType: "PLUS" | "LISTEN";
    score: number;
    createdAt: Date;
}

const ScoreSchema: Schema = new Schema({
    nickname: { type: String, required: true },
    gameType: { type: String, required: true, enum: ['PLUS', 'LISTEN'] },
    score: { type: Number, required: true },
    createdAt: { type: Date, default: Date.now }
});

export default mongoose.models.Score || mongoose.model<IScore>('Score', ScoreSchema);
