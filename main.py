import cv2
import time
import datetime
import numpy as np
import requests
import os
from dotenv import load_dotenv
import tensorflow as tf

load_dotenv()

# =========================
# 設定値
# =========================
INTERVAL = 12          # 推論間隔（秒）
WINDOW = 120           # 集計ウィンドウ（秒）
CONF_THRESHOLD = 0.6   # 信頼度しきい値
IGNORE_CLASS_ID = 0    # 「何もしてない」

# =========================
# TFLite モデル準備
# =========================
interpreter = tf.lite.Interpreter(
    model_path="./saved_model/game_classifier.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_MAP = {
    0: "何もしてない",
    1: "人生ゲーム",
    2: "スマブラ",
    3: "マリオカート",
}

# =========================
# カメラ準備
# =========================
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not capture.isOpened():
    print("❌ カメラが開けませんでした")
    exit()

# =========================
# API
# =========================
api_url = os.getenv("API_URL")
if not api_url:
    print("❌ API_URL が設定されていません")
    exit()

print("🎮 ゲーム推定開始（状態変化時のみ送信）")

results = []
window_start = time.time()
last_pred_time = 0

# ★ 前回送信したクラスID
last_sent_class_id = None

# =========================
# メインループ
# =========================
while True:
    ret, frame = capture.read()
    if not ret:
        time.sleep(0.1)
        continue

    now = time.time()

    # ---- interval ごとに推論 ----
    if now - last_pred_time >= INTERVAL:
        img = cv2.resize(frame, (128, 128))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])

        class_id = int(np.argmax(pred))
        confidence = float(np.max(pred))

        # ---- フィルタ条件 ----
        if (
            confidence >= CONF_THRESHOLD and
            class_id != IGNORE_CLASS_ID
        ):
            results.append((class_id, confidence))

        last_pred_time = now

    # ---- window 秒ごとに集計 ----
    if now - window_start >= WINDOW:
        if results:
            class_ids = [r[0] for r in results]
            most_common_id = max(set(class_ids), key=class_ids.count)

            max_conf = max(
                r[1] for r in results if r[0] == most_common_id
            )

            # ★ 前回と違うときだけ送信
            if most_common_id != last_sent_class_id:
                payload = {
                    "class_id": most_common_id,
                    "class_name": CLASS_MAP[most_common_id],
                    "confidence": round(max_conf, 3),
                    "timestamp": datetime.datetime.now().isoformat()
                }

                try:
                    requests.post(
                        api_url,
                        json=payload,
                        timeout=10
                    )
                    print(
                        f"📤 状態変化送信: {payload['class_name']} "
                        f"(conf={payload['confidence']})"
                    )
                    last_sent_class_id = most_common_id
                except Exception as e:
                    print("⚠️ API送信失敗:", e)
            else:
                print(
                    f"⏸ 同一状態継続中: {CLASS_MAP[most_common_id]}（送信なし）"
                )

        results.clear()
        window_start = now

    # ---- 完全ヘッドレス運用 ----
    time.sleep(0.01)
