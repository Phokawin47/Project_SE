require("dotenv").config();
const fs = require("fs");
const path = require("path");
const axios = require("axios");
const FormData = require("form-data");

const folderPath = "C:/Users/usEr/Documents/GitHub/Project_SE/SFX/Voice_1_To_55";

async function uploadFile(filePath) {
  const form = new FormData();
  form.append("audio", fs.createReadStream(filePath));

  try {
    const response = await axios.post("http://localhost:3000/upload", form, {
      headers: form.getHeaders(),
    });

    console.log(`Uploaded: ${path.basename(filePath)}`);
  } catch (error) {
    console.error(`Error uploading ${filePath}`);
  }
}

async function uploadAll() {
  const files = fs.readdirSync(folderPath);

  for (const file of files) {
    const fullPath = path.join(folderPath, file);

    if (fs.lstatSync(fullPath).isFile()) {
      await uploadFile(fullPath);
    }
  }

  console.log("All files uploaded.");
}

uploadAll();