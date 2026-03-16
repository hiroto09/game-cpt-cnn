import cv2
import time
import datetime
import numpy as np
import requests
import os
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

load_dotenv()

# ---- CNN モデルの準備 ----
model = load_model("./saved_model/game_classifier.h5")

# IDと日本語名の対応
CLASS_MAP = {
    0: "何もしてない",
    1: "人生ゲーム",
    2: "スマブラ",
    3: "マリオカート",
}

# ---- キャプチャーボードを開く ----
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("キャプチャーボードが開けませんでした")
    exit()

interval = 10
window = 130    
results = []
window_start = time.time()
last_pred_time = time.time()

# ---- APIエンドポイント ----
api_url = os.getenv("API_URL")
if not api_url:
    print("API_URL が設定されていません")
    exit()

print("🎮 ゲーム推定開始... (qで終了)")

while True:
    ret, frame = capture.read()
    if not ret:
        print("映像を取得できませんでした")
        break

    now = time.time()

    # intervalごとに推論
    if now - last_pred_time >= interval:
        img_resized = cv2.resize(frame, (128, 128))
        img_norm = img_resized / 255.0
        img_input = np.expand_dims(img_norm, axis=0)

        pred = model.predict(img_input, verbose=0)
        class_id = int(np.argmax(pred))
        confidence = float(np.max(pred))

        results.append((class_id, confidence))
        last_pred_time = now

    # window秒ごとに集計してAPI送信
    if now - window_start >= window and results:
        class_ids = [r[0] for r in results]
        most_common_id = max(set(class_ids), key=class_ids.count)
        max_conf = max(r[1] for r in results if r[0] == most_common_id)

        result = {
            "class_id": most_common_id,
            "confidence": max_conf,
            "timestamp": datetime.datetime.now().isoformat()
        }

        print("📡 API送信(JSON):", result)
        try:
            response = requests.post(
                api_url,
                json=result,   # ← JSON送信に変更
                timeout=10
            )
        except Exception as e:
            print("⚠️ API 送信エラー:", e)

        # リセット
        results = []
        window_start = now

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
