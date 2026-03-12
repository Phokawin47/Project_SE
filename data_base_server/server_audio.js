require("dotenv").config();
const express = require("express");
const { MongoClient, GridFSBucket } = require("mongodb");
const multer = require("multer");
const { Readable } = require("stream");

const app = express();
const uri = process.env.MONGO1_URI;

// 🔥 ต้องมีตัวนี้
const client = new MongoClient(uri);

let bucket;

const upload = multer({ storage: multer.memoryStorage() });

async function start() {
  await client.connect();

  const db = client.db("FunnyFingerAudioDB");

  bucket = new GridFSBucket(db, {
    bucketName: "SFXFiles"
  });

  console.log("Connected to MongoDB & GridFS Ready");

  // Upload + Save to GridFS
app.post("/upload", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    // 🔎 เช็คก่อนว่ามีไฟล์ชื่อเดียวกันหรือยัง
    const existingFile = await bucket.find({
      filename: req.file.originalname
    }).toArray();

    if (existingFile.length > 0) {
      return res.status(409).json({
        message: "File already exists",
        filename: req.file.originalname
      });
    }

    const readableStream = Readable.from(req.file.buffer);

    const uploadStream = bucket.openUploadStream(req.file.originalname, {
      contentType: req.file.mimetype
    });

    readableStream.pipe(uploadStream)
      .on("error", () => res.status(500).json({ error: "Upload error" }))
      .on("finish", () => {
        res.json({
          message: "Upload success",
          fileId: uploadStream.id
        });
      });

  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
});

  // Stream audio
  app.get("/audio/:filename", async (req, res) => {
    try {
      const files = await bucket.find({ filename: req.params.filename }).toArray();

      if (!files || files.length === 0) {
        return res.status(404).json({ error: "File not found" });
      }

      res.set("Content-Type", files[0].contentType || "audio/mpeg");

      const downloadStream = bucket.openDownloadStreamByName(req.params.filename);
      downloadStream.pipe(res);

    } catch (err) {
      res.status(500).json({ error: "Server error" });
    }
  });

  app.listen(3000, () => {
    console.log("Server running on port 3000");
  });
}

start().catch(console.error);