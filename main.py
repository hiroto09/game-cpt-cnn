import cv2
import time
import datetime
import numpy as np
import requests
import os
from dotenv import load_dotenv
import tensorflow as tf

load_dotenv()

# ---- TFLite モデルの準備 ----
interpreter = tf.lite.Interpreter(model_path="./saved_model/game_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# IDと日本語名の対応
CLASS_MAP = {
    0: "何もしてない",
    1: "人生ゲーム",
    2: "スマブラ",
    3: "マリオカート",
}

# ---- カメラを開く ----
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not capture.isOpened():
    print("カメラが開けませんでした")
    exit()

interval = 12       # 推論間隔（秒）
window = 120        # 集計ウィンドウ（秒）
results = []

window_start = time.time()
last_pred_time = 0

# ---- APIエンドポイント ----
api_url = os.getenv("API_URL")
if not api_url:
    print("API_URL が設定されていません")
    exit()

print("🎮 ゲーム推定開始（Raspberry Pi） qで終了")

while True:
    ret, frame = capture.read()
    if not ret:
        time.sleep(0.1)
        continue

    now = time.time()

    # ---- interval ごとに推論 ----
    if now - last_pred_time >= interval:
        img = cv2.resize(frame, (128, 128))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])

        class_id = int(np.argmax(pred))
        confidence = float(np.max(pred))

        results.append((class_id, confidence))
        last_pred_time = now


    # ---- window 秒ごとに集計して API 送信 ----
    if now - window_start >= window and results:
        class_ids = [r[0] for r in results]
        most_common_id = max(set(class_ids), key=class_ids.count)
        max_conf = max(r[1] for r in results if r[0] == most_common_id)

        payload = {
            "class_id": most_common_id,
            "confidence": max_conf,
            "timestamp": datetime.datetime.now().isoformat()
        }

        _, img_encoded = cv2.imencode(".jpg", frame)


        try:
            requests.post(
                api_url,
                data=payload,
                files={
                    "image": (
                        "latest_frame.jpg",
                        img_encoded.tobytes(),
                        "image/jpeg"
                    )
                },
                timeout=10
            )
        except Exception as e:
            print("⚠️ API送信失敗:", e)

        results.clear()
        window_start = now

    # ---- 表示（不要なら消してOK）----
    cv2.imshow("Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
