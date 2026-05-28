# test_stream_FFMPEG.py
import subprocess
import numpy as np
import cv2
import time

WIDTH = 704
HEIGHT = 576

FFMPEG_PATH = r"C:\Users\idf32345\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
RTSP_URL = "rtsp://admin:Suzuki0%21@172.16.224.72:554/cam/realmonitor?channel=1&subtype=1"

ffmpeg_cmd = [
    FFMPEG_PATH,
    "-rtsp_transport", "tcp",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-i", RTSP_URL,
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]

pipe = subprocess.Popen(
    ffmpeg_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,   # JANGAN DEVNULL dulu, biar error FFmpeg kelihatan
    bufsize=10**8
)

prev_time = time.time()
frame_size = WIDTH * HEIGHT * 3

while True:
    raw = pipe.stdout.read(frame_size)
    if len(raw) != frame_size:
        print("⚠️ Frame tidak lengkap / stream putus")
        # cek pesan error FFmpeg
        err = pipe.stderr.read(2000)
        if err:
            print(err.decode(errors="ignore"))
        break

    frame = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3))

    now = time.time()
    fps = 1 / (now - prev_time + 1e-6)
    prev_time = now

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("RTSP TEST", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pipe.terminate()
cv2.destroyAllWindows()