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
WIDTH = 704
HEIGHT = 576
YOLO_INTERVAL = 15
OCR_INTERVAL = 15
MIN_VOTES = 5
MIN_RATIO = 0.4

ORDS_URL = "http://idtbecsapexdev/ecs/ecs/receive/insert"

FFMPEG_PATH = r"C:\Users\idf32345\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
RTSP_URL = "rtsp://admin:Suzuki0%21@172.16.224.72:554/cam/realmonitor?channel=1&subtype=1"

# ==============================
# GUI LAYOUT CONFIG
# ==============================
PANEL_W = 420
LOG_MAX_ROWS = 12
CANVAS_W = WIDTH + PANEL_W
CANVAS_H = max(HEIGHT, 80 + LOG_MAX_ROWS * 32 + 60)

# Warna (BGR)
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
ords_sending    = set()   # plate sedang dikirim (thread aktif)

# ==============================
# FFmpeg CMD (reusable untuk reconnect)
# ==============================
ffmpeg_cmd = [
    FFMPEG_PATH,
    "-hwaccel", "cuda",
    "-rtsp_transport", "tcp",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-i", RTSP_URL,
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]

def start_ffmpeg():
    print("🔌 Menghubungkan ke RTSP...")
    return subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )

pipe = start_ffmpeg()

# ==============================
# VIDEO OUTPUT
# ==============================
ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
out = cv2.VideoWriter(
    f"hasil_inference_rtsp_ffmpeg_{ts_str}.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
    (WIDTH, HEIGHT)
)

prev_time = time.time()

# ==============================
# UTIL FUNCTIONS
# ==============================
def image_to_base64(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok:
        b64 = base64.b64encode(buf).decode()
        print(f"📸 Base64 length: {len(b64)}")
        return b64
    return None


def insert_plate_to_ords(plate, img):
    global ords_status, ords_last_ms

    b64_img = image_to_base64(img)
    if not b64_img:
        print("❌ Failed to encode image")
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
        r = requests.post(
            ORDS_URL,
            json=payload,
            timeout=20,
            headers={"Content-Type": "application/json"}
        )
        ords_last_ms = int((time.time() - t0) * 1000)
        print(f"🌐 ORDS {r.status_code}: {plate}")

        if r.text:
            try:
                resp_json = r.json()
                msg = resp_json.get("message") or resp_json.get("msg") or resp_json.get("error")
                if msg:
                    print(f"🧾 ORDS MESSAGE: {msg}")
                else:
                    print("✅ ORDS RESPONSE JSON:")
                    print(json.dumps(resp_json, indent=2, ensure_ascii=False))
            except Exception:
                print("✅ ORDS RESPONSE TEXT:")
                print(r.text)

        if r.status_code not in (200, 201):
            print("❌ ORDS FAILED")
            ords_status = "FAIL"
            return False

        print("✅ ORDS SUCCESS")
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
# DRAW PANEL (sisi kanan)
# ==============================
def draw_panel(canvas, fps, vote_info):
    px = WIDTH

    cv2.rectangle(canvas, (px, 0), (CANVAS_W, CANVAS_H), C_BG, -1)

    # Header
    cv2.rectangle(canvas, (px, 0), (CANVAS_W, 50), C_HEADER, -1)
    cv2.putText(canvas, "ANPR SYSTEM", (px + 10, 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, C_CYAN, 2)

    y = 65
    # Status bar
    cv2.putText(canvas, f"FPS: {fps:.1f}", (px + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_YELLOW, 2)
    now_str = datetime.now().strftime("%H:%M:%S")
    cv2.putText(canvas, now_str, (px + 130, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1)

    # ORDS indicator
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

    # Voting progress
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

    # Log table
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

    # Footer
    cv2.rectangle(canvas, (px, CANVAS_H - 28), (CANVAS_W, CANVAS_H), C_HEADER, -1)
    footer_txt = f"Locked: {len(plate_locked)}  |  Votes pool: {sum(len(v) for v in plate_votes.values())}"
    cv2.putText(canvas, footer_txt, (px + 10, CANVAS_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1)


# ==============================
# MAIN LOOP
# ==============================
frame_with_overlay = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
reconnect_count = 0

print(f"📐 Canvas: {CANVAS_W}x{CANVAS_H}")
print("🚀 ANPR dimulai — tekan Q untuk keluar")

while True:
    try:
        raw = pipe.stdout.read(WIDTH * HEIGHT * 3)
    except Exception as e:
        print(f"❌ Pipe read error: {e}")
        raw = None

    # ── AUTO RECONNECT jika pipe EOF ──
    if not raw:
        reconnect_count += 1
        rc = pipe.poll()
        print(f"⚠️ FFmpeg EOF (return code={rc}) — reconnect #{reconnect_count} dalam 3 detik...")
        try:
            pipe.terminate()
        except Exception:
            pass
        time.sleep(3)
        pipe = start_ffmpeg()
        # Tunggu sebentar biar FFmpeg ready
        time.sleep(1)
        continue

    frame = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3))
    frame_id += 1

    frame_small = cv2.resize(frame, (640, 360))
    sx, sy = WIDTH / 640, HEIGHT / 360

    # ── YOLO ──
    if frame_id % YOLO_INTERVAL == 0:
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

    # ── OCR + VOTING ──
    if frame_id % OCR_INTERVAL == 0:
        current_box_keys = set(last_boxes)
        for k in [k for k in last_ocr_result if k not in current_box_keys]:
            del last_ocr_result[k]

        for x1, y1, x2, y2 in last_boxes:
            try:
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue

                print(f"🔍 OCR box ({x1},{y1},{x2},{y2}) size={x2-x1}x{y2-y1}") 

                plate, conf = read_license_plate(crop)

                if not plate:
                    continue
                if not license_complies_format(plate):
                    continue
                if plate in plate_locked:
                    continue

                last_ocr_result[(x1, y1, x2, y2)] = (plate, conf)
                plate_votes[plate].append(time.time())
                print(f"🗳️ VOTE {plate}: {len(plate_votes[plate])}")

                final_plate = get_final_plate()
                if final_plate and final_plate not in ords_sending:
                    print(f"🎯 FINAL PLATE: {final_plate}")
                    snap = frame_with_overlay.copy()

                    # Lock dulu sebelum thread jalan agar tidak double-send
                    plate_locked.add(final_plate)
                    ords_sending.add(final_plate)
                    plate_votes.clear()
                    last_ocr_result.clear()

                    def send_async(p, img, t):
                        success = insert_plate_to_ords(p, img)
                        log_entries.append({
                            "time":   t,
                            "plate":  p,
                            "status": "OK" if success else "FAIL"
                        })
                        ords_sending.discard(p)
                        if not success:
                            # ORDS gagal: unlock agar bisa retry
                            plate_locked.discard(p)
                            print(f"⚠️ ORDS FAIL — {p} di-unlock untuk retry")
                        else:
                            print(f"✅ INSERTED & LOCKED: {p}")

                    t = threading.Thread(
                        target=send_async,
                        args=(final_plate, snap, datetime.now().strftime("%H:%M:%S")),
                        daemon=True
                    )
                    t.start()

            except Exception as e:
                print(f"❌ OCR error box ({x1},{y1},{x2},{y2}): {e}")
                import traceback
                traceback.print_exc()
                continue

    # ── DRAW frame kamera ──
    for x1, y1, x2, y2 in last_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_BOX_PLATE, 2)
        ocr_data = last_ocr_result.get((x1, y1, x2, y2))
        if ocr_data:
            plate_text, conf = ocr_data
            cv2.putText(frame, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_TEXT_PLATE, 2, cv2.LINE_AA)
            cv2.putText(frame, f"OCR: {conf:.2f}", (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_YELLOW, 1, cv2.LINE_AA)
            cv2.putText(frame, f"W:{x2-x1}px", (x1, y1 - 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1)

    frame_with_overlay = frame.copy()

    # ── COMPOSE CANVAS ──
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:HEIGHT, :WIDTH] = frame

    now = time.time()
    fps = 1 / (now - prev_time + 1e-6)
    prev_time = now
    cv2.putText(canvas, f"FPS: {fps:.1f}", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_CYAN, 2)

    vote_info = sorted(
        [(p, len(t)) for p, t in plate_votes.items()],
        key=lambda x: -x[1]
    )[:4]

    draw_panel(canvas, fps, vote_info)

    cv2.imshow("ANPR RTSP", canvas)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================
print("🛑 Program dihentikan")
pipe.terminate()
out.release()
cv2.destroyAllWindows()