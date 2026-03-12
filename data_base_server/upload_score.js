const axios = require("axios");

async function uploadScore() {
  try {
    const response = await axios.post(
      "http://localhost:4000/score",
      {
        player: "Player1",
        score: 120
      }
    );
    console.log("Response:", response.data);
  } catch (error) {
    console.error("Upload failed");
    if (error.response) {
      console.error(error.response.data);
    } else {
      console.error(error.message);
    }
  }
}

uploadScore();