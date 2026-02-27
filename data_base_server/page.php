<?php
$filename = "0.mp3"; // ชื่อไฟล์ที่อยู่ใน MongoDB
?>

<!DOCTYPE html>
<html>
<head>
    <title>Audio Test</title>
</head>
<body>

<h2>ทดสอบเล่นไฟล์เสียง</h2>

<audio controls>
    <source src="http://localhost:3000/audio/<?php echo $filename; ?>" type="audio/mpeg">
    Your browser does not support the audio element.
</audio>

</body>
</html>