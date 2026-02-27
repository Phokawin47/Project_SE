async function loadAudio(filename) {

  const res = await fetch(
    "http://localhost:3000/get-audio/" + filename
  );

  const data = await res.json();

  document.getElementById("player").src = data.url;
}