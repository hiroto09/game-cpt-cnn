import cv2
import time
import datetime
import numpy as np
import requests
import os
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# =========================
# 環境変数
# =========================
load_dotenv()
api_url = os.getenv("API_URL")

if not api_url:
    print("❌ API_URL が設定されていません")
    exit()

# =========================
# モデル読み込み
# =========================
model = load_model("./saved_model/game_classifier.h5")

CLASS_MAP = {
    0: "何もしてない",
    1: "人生ゲーム",
    2: "スマブラ",
    3: "マリオカート",
}

# =========================
# カメラ初期化関数（重要）
# =========================
def open_camera():
    for i in range(3):  # 0,1,2を試す
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ カメラ{i}で接続")
            return cap
    return None

capture = open_camera()

if capture is None:
    print("❌ カメラが見つかりません")
    exit()

# =========================
# 設定
# =========================
interval = 10      # 推論間隔（秒）
window = 130       # 集計時間（秒）

results = []
window_start = time.time()
last_pred_time = time.time()

print("🎮 ゲーム推定開始... (qで終了)")

# =========================
# メインループ
# =========================
while True:
    ret, frame = capture.read()

    # =========================
    # 🔥 映像取得失敗 → 自動復旧
    # =========================
    if not ret:
        print("⚠️ 映像取得失敗 → 再接続中...")

        capture.release()
        time.sleep(2)

        capture = open_camera()

        if capture is None:
            print("❌ 再接続失敗...リトライ")
            time.sleep(2)

        continue

    now = time.time()

    # =========================
    # 推論処理
    # =========================
    if now - last_pred_time >= interval:
        img_resized = cv2.resize(frame, (128, 128))
        img_norm = img_resized / 255.0
        img_input = np.expand_dims(img_norm, axis=0)

        pred = model.predict(img_input, verbose=0)
        class_id = int(np.argmax(pred))
        confidence = float(np.max(pred))

        print(f"推定: {CLASS_MAP[class_id]} ({confidence:.2f})")

        results.append((class_id, confidence))
        last_pred_time = now

    # =========================
    # 集計してAPI送信
    # =========================
    if now - window_start >= window and results:
        class_ids = [r[0] for r in results]
        most_common_id = max(set(class_ids), key=class_ids.count)
        max_conf = max(r[1] for r in results if r[0] == most_common_id)

        result = {
            "class_id": most_common_id,
            "confidence": max_conf,
            "timestamp": datetime.datetime.now().isoformat()
        }

        print("📡 API送信:", result)

        try:
            response = requests.post(api_url, json=result, timeout=10)
            print("✅ 送信成功:", response.status_code)
        except Exception as e:
            print("⚠️ API送信エラー:", e)

        # リセット
        results = []
        window_start = now

    # =========================
    # 終了キー
    # =========================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # CPU負荷軽減
    time.sleep(0.01)

# =========================
# 終了処理
# =========================
capture.release()
cv2.destroyAllWindows()