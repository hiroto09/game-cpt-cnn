import cv2
import time
import datetime
import numpy as np
import requests
import os
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

load_dotenv()

# =========================
# CNN モデルの準備
# =========================
model = load_model("./saved_model/game_classifier.h5")

# IDと日本語名の対応
CLASS_MAP = {
    0: "何もしてない",
    1: "人生ゲーム",
    2: "スマブラ",
    3: "マリオカート",
}

# =========================
# キャプチャーボード
# =========================
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("❌ キャプチャーボードが開けませんでした")
    exit()

# =========================
# 設定値
# =========================
INTERVAL = 12          # 推論間隔（秒）
WINDOW = 120           # 集計ウィンドウ（秒）

results = []
window_start = time.time()
last_pred_time = time.time()

# =========================
# API
# =========================
api_url = os.getenv("API_URL")
if not api_url:
    print("❌ API_URL が設定されていません")
    exit()

print("🎮 ゲーム推定開始... (qで終了)")

# =========================
# メインループ
# =========================
while True:
    ret, frame = capture.read()
    if not ret:
        print("⚠️ 映像を取得できませんでした")
        break

    now = time.time()

    # ---- interval ごとに推論 ----
    if now - last_pred_time >= INTERVAL:
        img_resized = cv2.resize(frame, (128, 128))
        img_norm = img_resized / 255.0
        img_input = np.expand_dims(img_norm, axis=0)

        pred = model.predict(img_input, verbose=0)
        class_id = int(np.argmax(pred))
        confidence = float(np.max(pred))

        results.append((class_id, confidence))
        last_pred_time = now

    # ---- window 秒ごとに集計して API 送信 ----
    if now - window_start >= WINDOW and results:
        class_ids = [r[0] for r in results]
        most_common_id = max(set(class_ids), key=class_ids.count)
        max_conf = max(r[1] for r in results if r[0] == most_common_id)

        payload = {
            "class_id": most_common_id,
            "class_name": CLASS_MAP.get(most_common_id, "unknown"),
            "confidence": round(max_conf, 3),
            "timestamp": datetime.datetime.now().isoformat()
        }

        print("📡 API送信:", payload)
        try:
            requests.post(
                api_url,
                data=payload,   # ← 画像なし・formデータのみ
                timeout=10
            )
        except Exception as e:
            print("⚠️ API送信エラー:", e)

        # リセット
        results.clear()
        window_start = now

    # ---- 表示（不要なら丸ごと消してOK）----
    cv2.imshow("Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# 後処理
# =========================
capture.release()
cv2.destroyAllWindows()
