require("dotenv").config();
const express = require("express");
const { MongoClient } = require("mongodb");

const app = express();
app.use(express.json());

const uri = process.env.MONGO2_URI;
const client = new MongoClient(uri);

let scoreCollection;

async function start() {
  await client.connect();
  const db = client.db("gameScoreDB");
  scoreCollection = db.collection("scores");
  console.log("Connected to gameScoreDB");
  app.listen(4000, () => {
    console.log("Score server running on port 4000");
  });
}
start().catch(console.error);

// =====================
// Upload Score
// =====================
app.post("/score", async (req, res) => {
  try {
    const { player, score } = req.body;
    if (!player || score === undefined) {
      return res.status(400).json({
        error: "player and score required"
      });
    }
    const result = await scoreCollection.insertOne({
      player,
      score,
      createdAt: new Date()
    });
    res.json({
      message: "Score saved",
      id: result.insertedId
    });
  } catch (err) {
    res.status(500).json({
      error: "Server error"
    });
  }
});
// =====================
// Leaderboard
// =====================
app.get("/leaderboard", async (req, res) => {
  try {
    const topScores = await scoreCollection
      .find()
      .sort({ score: -1 })
      .limit(10)
      .toArray();
    res.json(topScores);
  } catch (err) {
    res.status(500).json({
      error: "Server error"
    });
  }
});