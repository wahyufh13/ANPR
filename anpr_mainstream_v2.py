import subprocess
import numpy as np
import cv2
import time
import json
import requests
from datetime import datetime
from ultralytics import YOLO
from util_FFMPEG import read_license_plate, license_complies_format
from collections import defaultdict
import base64
import threading

# ==============================
# CONFIG
# ==============================
YOLO_INTERVAL = 15
OCR_INTERVAL = 15
MIN_VOTES = 5
MIN_RATIO = 0.4

ORDS_URL = "http://idtbecsapexdev/ecs/ecs/receive/insert"

FFMPEG_PATH = r"C:\Users\idf32345\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
FFPROBE_PATH = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")

RTSP_URL = "rtsp://admin:Suzuki0%21@172.16.157.201:554/cam/realmonitor?channel=1&subtype=0"

# Toggle untuk rekaman (matikan kalau mau FPS maksimal)
ENABLE_RECORDING = False

# ==============================
# AUTO-DETECT RESOLUSI
# ==============================
def detect_resolution(rtsp_url):
    print("🔎 Mendeteksi resolusi kamera...")
    cmd = [
        FFPROBE_PATH, "-v", "error", "-rtsp_transport", "tcp",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", rtsp_url
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        info = json.loads(result.stdout)
        w = info["streams"][0]["width"]
        h = info["streams"][0]["height"]
        print(f"✅ Resolusi terdeteksi: {w}x{h}")
        return w, h
    except Exception as e:
        print(f"⚠️ Gagal deteksi resolusi: {e}")
        return 1920, 1080

WIDTH, HEIGHT = detect_resolution(RTSP_URL)

MAX_DISPLAY_W = 1280
display_scale = min(1.0, MAX_DISPLAY_W / WIDTH)
DISP_W = int(WIDTH * display_scale)
DISP_H = int(HEIGHT * display_scale)
print(f"📺 Display scale: {display_scale:.2f} → {DISP_W}x{DISP_H}")

# ==============================
# GUI LAYOUT
# ==============================
PANEL_W = 420
LOG_MAX_ROWS = 12
CANVAS_W = DISP_W + PANEL_W
CANVAS_H = max(DISP_H, 80 + LOG_MAX_ROWS * 32 + 60)

C_BG         = (30,  30,  30)
C_HEADER     = (50,  50,  50)
C_GREEN      = (0,   200, 80)
C_RED        = (0,   60,  220)
C_YELLOW     = (0,   210, 255)
C_CYAN       = (220, 210, 0)
C_WHITE      = (240, 240, 240)
C_GRAY       = (130, 130, 130)
C_ORANGE     = (0,   150, 255)
C_BOX_PLATE  = (0,   0,   220)
C_TEXT_PLATE = (0,   230, 0)

# ==============================
# LOAD YOLO
# ==============================
model = YOLO(
    r"C:\Users\idf32345\Documents\001 Computer Vision\automatic-number-plate-recognition-python-yolov8-main\best.pt"
)
model.to("cuda")
print("✅ YOLO model OK")

# ==============================
# GLOBAL STATE
# ==============================
frame_id        = 0
last_boxes      = []
plate_votes     = defaultdict(list)
plate_locked    = set()
last_ocr_result = {}
log_entries     = []
ords_status     = "IDLE"
ords_last_ms    = 0
ords_sending    = set()

# ==============================
# 🔥 FRAME GRABBER THREAD (KUNCI SOLUSI DELAY)
# ==============================
class FrameGrabber:
    """
    Thread terpisah untuk baca pipe FFmpeg terus-menerus.
    Main loop selalu dapat frame TERBARU (yang lama dibuang).
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.pipe = None
        self.reconnect_count = 0
        self.last_frame_time = time.time()
        self.dropped_frames = 0
        self.total_frames = 0

    def _ffmpeg_cmd(self):
        return [
            FFMPEG_PATH,
            "-hwaccel", "cuda",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-strict", "experimental",
            "-avioflags", "direct",
            "-rtbufsize", "1M",       # ← buffer kecil, jangan menumpuk
            "-i", RTSP_URL,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",                    # disable audio
            "-sn",                    # disable subtitle
            "-"
        ]

    def _start_pipe(self):
        print(f"🔌 Menghubungkan ke RTSP... (reconnect #{self.reconnect_count})")
        self.pipe = subprocess.Popen(
            self._ffmpeg_cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.width * self.height * 3 * 2  # buffer cukup 2 frame saja
        )

    def start(self):
        self._start_pipe()
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        frame_size = self.width * self.height * 3
        while self.running:
            try:
                raw = self.pipe.stdout.read(frame_size)
                if not raw or len(raw) < frame_size:
                    # EOF / corrupt → reconnect
                    self.reconnect_count += 1
                    print(f"⚠️ Pipe EOF — reconnect #{self.reconnect_count}")
                    try:
                        self.pipe.terminate()
                    except Exception:
                        pass
                    time.sleep(2)
                    self._start_pipe()
                    continue

                frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))
                self.total_frames += 1
                
                with self.lock:
                    if self.frame is not None:
                        # Frame lama belum sempat dipakai → drop counter
                        self.dropped_frames += 1
                    self.frame = frame
                    self.last_frame_time = time.time()

            except Exception as e:
                print(f"❌ Grabber error: {e}")
                time.sleep(1)

    def read(self):
        """Ambil frame terbaru, lalu null-kan supaya main loop tau frame sudah dikonsumsi."""
        with self.lock:
            frame = self.frame
            self.frame = None
        return frame

    def stop(self):
        self.running = False
        try:
            self.pipe.terminate()
        except Exception:
            pass

# Start grabber
grabber = FrameGrabber(WIDTH, HEIGHT)
grabber.start()
time.sleep(2)  # tunggu frame pertama

# ==============================
# VIDEO OUTPUT (optional)
# ==============================
out = None
if ENABLE_RECORDING:
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = cv2.VideoWriter(
        f"hasil_inference_rtsp_ffmpeg_{ts_str}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        15,
        (DISP_W, DISP_H)
    )

prev_time = time.time()

# ==============================
# UTIL FUNCTIONS
# ==============================
def image_to_base64(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok:
        return base64.b64encode(buf).decode()
    return None


def insert_plate_to_ords(plate, img):
    global ords_status, ords_last_ms
    b64_img = image_to_base64(img)
    if not b64_img:
        ords_status = "FAIL"
        return False

    payload = {
        "p_number_plate": plate,
        "p_gate_type": "IN",
        "p_image_base64": b64_img
    }

    ords_status = "SENDING"
    t0 = time.time()
    try:
        r = requests.post(ORDS_URL, json=payload, timeout=20,
                          headers={"Content-Type": "application/json"})
        ords_last_ms = int((time.time() - t0) * 1000)
        print(f"🌐 ORDS {r.status_code}: {plate}")
        
        if r.status_code not in (200, 201):
            ords_status = "FAIL"
            return False
        ords_status = "OK"
        return True
    except Exception as e:
        ords_last_ms = int((time.time() - t0) * 1000)
        print("❌ ORDS ERROR:", e)
        ords_status = "FAIL"
        return False


def get_final_plate():
    total = sum(len(v) for v in plate_votes.values())
    if total == 0:
        return None
    for plate, times in plate_votes.items():
        if len(times) >= MIN_VOTES:
            ratio = len(times) / total
            if ratio >= MIN_RATIO:
                return plate
    return None


# ==============================
# 🔥 OCR di THREAD TERPISAH (KUNCI #2)
# ==============================
ocr_queue = []
ocr_queue_lock = threading.Lock()
ocr_busy = False

def ocr_worker():
    """Thread khusus untuk OCR supaya tidak blocking main loop."""
    global ocr_busy
    while True:
        job = None
        with ocr_queue_lock:
            if ocr_queue:
                job = ocr_queue.pop(0)
                ocr_busy = True
        
        if job is None:
            time.sleep(0.01)
            continue
        
        crop, box_key, frame_snapshot = job
        try:
            plate, conf = read_license_plate(crop)
            if plate and license_complies_format(plate) and plate not in plate_locked:
                last_ocr_result[box_key] = (plate, conf)
                plate_votes[plate].append(time.time())
                print(f"🗳️ VOTE {plate}: {len(plate_votes[plate])}")
                
                final_plate = get_final_plate()
                if final_plate and final_plate not in ords_sending:
                    print(f"🎯 FINAL PLATE: {final_plate}")
                    plate_locked.add(final_plate)
                    ords_sending.add(final_plate)
                    plate_votes.clear()
                    last_ocr_result.clear()
                    
                    def send_async(p, img, t):
                        success = insert_plate_to_ords(p, img)
                        log_entries.append({
                            "time": t, "plate": p,
                            "status": "OK" if success else "FAIL"
                        })
                        ords_sending.discard(p)
                        if not success:
                            plate_locked.discard(p)
                    
                    threading.Thread(
                        target=send_async,
                        args=(final_plate, frame_snapshot, datetime.now().strftime("%H:%M:%S")),
                        daemon=True
                    ).start()
        except Exception as e:
            print(f"❌ OCR worker error: {e}")
        finally:
            with ocr_queue_lock:
                ocr_busy = len(ocr_queue) > 0

threading.Thread(target=ocr_worker, daemon=True).start()
print("✅ OCR worker thread started")

# ==============================
# DRAW PANEL
# ==============================
def draw_panel(canvas, fps, vote_info, lag_info):
    px = DISP_W
    cv2.rectangle(canvas, (px, 0), (CANVAS_W, CANVAS_H), C_BG, -1)
    cv2.rectangle(canvas, (px, 0), (CANVAS_W, 50), C_HEADER, -1)
    cv2.putText(canvas, "ANPR SYSTEM", (px + 10, 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, C_CYAN, 2)

    y = 65
    cv2.putText(canvas, f"FPS: {fps:.1f}", (px + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_YELLOW, 2)
    cv2.putText(canvas, datetime.now().strftime("%H:%M:%S"), (px + 130, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1)

    if ords_status == "OK":
        s_color, s_label = C_GREEN, f"ORDS OK  {ords_last_ms}ms"
    elif ords_status == "FAIL":
        s_color, s_label = C_RED, "ORDS FAIL"
    elif ords_status == "SENDING":
        s_color, s_label = C_YELLOW, "ORDS ..."
    else:
        s_color, s_label = C_GRAY, "ORDS IDLE"

    cv2.circle(canvas, (px + 260, y - 6), 7, s_color, -1)
    cv2.putText(canvas, s_label, (px + 276, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_color, 1)

    y += 28
    cv2.line(canvas, (px + 5, y), (CANVAS_W - 5, y), C_GRAY, 1)

    # Resolusi + lag info
    y += 18
    cv2.putText(canvas, f"SOURCE: {WIDTH}x{HEIGHT}", (px + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1)
    y += 16
    cv2.putText(canvas, lag_info, (px + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1)
    y += 8
    cv2.line(canvas, (px + 5, y), (CANVAS_W - 5, y), C_GRAY, 1)

    # Voting
    y += 18
    cv2.putText(canvas, "VOTING PROGRESS", (px + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GRAY, 1)
    y += 22

    if vote_info:
        for plate_text, count in vote_info:
            pct = min(count / MIN_VOTES, 1.0)
            bar_w = int(pct * (PANEL_W - 160))
            bar_color = C_GREEN if pct >= 1.0 else C_ORANGE
            cv2.putText(canvas, f"{plate_text}", (px + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 1)
            cv2.rectangle(canvas, (px + 155, y - 14), (px + 155 + (PANEL_W - 160), y + 2), (60, 60, 60), -1)
            cv2.rectangle(canvas, (px + 155, y - 14), (px + 155 + bar_w, y + 2), bar_color, -1)
            cv2.putText(canvas, f"{count}/{MIN_VOTES}", (px + 160 + (PANEL_W - 160), y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GRAY, 1)
            y += 26
    else:
        cv2.putText(canvas, "Menunggu deteksi...", (px + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GRAY, 1)
        y += 26

    y += 8
    cv2.line(canvas, (px + 5, y), (CANVAS_W - 5, y), C_GRAY, 1)

    # Log
    y += 18
    cv2.putText(canvas, "LOG MASUK", (px + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GRAY, 1)
    y += 6

    col_time   = px + 10
    col_plate  = px + 90
    col_status = px + 260

    y += 20
    cv2.rectangle(canvas, (px, y - 16), (CANVAS_W, y + 6), (55, 55, 55), -1)
    cv2.putText(canvas, "WAKTU",  (col_time,   y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GRAY, 1)
    cv2.putText(canvas, "PLAT",   (col_plate,  y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GRAY, 1)
    cv2.putText(canvas, "STATUS", (col_status, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GRAY, 1)
    y += 8
    cv2.line(canvas, (px + 5, y), (CANVAS_W - 5, y), C_GRAY, 1)

    for i, entry in enumerate(reversed(log_entries[-LOG_MAX_ROWS:])):
        y += 28
        row_color = (40, 40, 40) if i % 2 == 0 else (48, 48, 48)
        cv2.rectangle(canvas, (px, y - 18), (CANVAS_W, y + 8), row_color, -1)
        cv2.putText(canvas, entry["time"],   (col_time,   y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_WHITE,  1)
        cv2.putText(canvas, entry["plate"],  (col_plate,  y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_YELLOW, 2)
        st_color = C_GREEN if entry["status"] == "OK" else C_RED
        cv2.putText(canvas, entry["status"], (col_status, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, st_color, 1)

    cv2.rectangle(canvas, (px, CANVAS_H - 28), (CANVAS_W, CANVAS_H), C_HEADER, -1)
    footer_txt = f"Locked: {len(plate_locked)}  |  Votes: {sum(len(v) for v in plate_votes.values())}"
    cv2.putText(canvas, footer_txt, (px + 10, CANVAS_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1)


# ==============================
# MAIN LOOP
# ==============================
frame_with_overlay = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
YOLO_W, YOLO_H = 640, 360
sx = WIDTH / YOLO_W
sy = HEIGHT / YOLO_H

print(f"📐 Frame: {WIDTH}x{HEIGHT} | YOLO: {YOLO_W}x{YOLO_H} | Display: {DISP_W}x{DISP_H}")
print("🚀 ANPR dimulai — tekan Q untuk keluar")

last_panel_update = 0
cached_panel_data = (0.0, [], "")

while True:
    # 🔥 Ambil frame TERBARU dari grabber (drop frame lama otomatis)
    frame = grabber.read()
    if frame is None:
        time.sleep(0.005)
        continue

    frame_id += 1

    # ── YOLO ──
    if frame_id % YOLO_INTERVAL == 0:
        frame_small = cv2.resize(frame, (YOLO_W, YOLO_H))
        try:
            results = model(frame_small, verbose=False)[0]
            last_boxes.clear()
            for x1, y1, x2, y2, score, cls in results.boxes.data.tolist():
                if score < 0.3:
                    continue
                last_boxes.append((
                    int(x1 * sx), int(y1 * sy),
                    int(x2 * sx), int(y2 * sy)
                ))
        except Exception as e:
            print(f"❌ YOLO error: {e}")

    # ── Submit OCR job ke worker thread (NON-BLOCKING) ──
    if frame_id % OCR_INTERVAL == 0:
        with ocr_queue_lock:
            # Kosongkan queue OCR lama biar tidak menumpuk
            ocr_queue.clear()
        
        for x1, y1, x2, y2 in last_boxes:
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            with ocr_queue_lock:
                ocr_queue.append((crop, (x1, y1, x2, y2), frame.copy()))

    # ── DRAW ──
    for x1, y1, x2, y2 in last_boxes:
        thickness = max(2, int(WIDTH / 800))
        font_scale = WIDTH / 1500
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_BOX_PLATE, thickness)
        ocr_data = last_ocr_result.get((x1, y1, x2, y2))
        if ocr_data:
            plate_text, conf = ocr_data
            cv2.putText(frame, plate_text, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, C_TEXT_PLATE, thickness, cv2.LINE_AA)

    frame_with_overlay = frame.copy()

    # Resize untuk display
    if display_scale < 1.0:
        frame_display = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)
    else:
        frame_display = frame

    # Compose canvas
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:DISP_H, :DISP_W] = frame_display

    now = time.time()
    fps = 1 / (now - prev_time + 1e-6)
    prev_time = now

    # 🔥 Update panel cuma 2x/detik (hemat CPU)
    if now - last_panel_update > 0.5:
        vote_info = sorted(
            [(p, len(t)) for p, t in plate_votes.items()],
            key=lambda x: -x[1]
        )[:4]
        lag_info = f"Drop: {grabber.dropped_frames}/{grabber.total_frames} | OCR Q: {len(ocr_queue)}"
        cached_panel_data = (fps, vote_info, lag_info)
        last_panel_update = now

    draw_panel(canvas, *cached_panel_data)

    cv2.imshow("ANPR RTSP", canvas)
    if out is not None:
        out.write(frame_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================
print("🛑 Program dihentikan")
print(f"📊 Total frames: {grabber.total_frames} | Dropped: {grabber.dropped_frames}")
grabber.stop()
if out is not None:
    out.release()
cv2.destroyAllWindows()
