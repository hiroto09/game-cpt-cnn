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
class_names = ["何もしてない","人生ゲーム", "スマブラ"]  # dataset のフォルダ名に合わせる

# ---- キャプチャーボードを開く ----
capture = cv2.VideoCapture(0)  # 環境に応じて 0,1,2 を変更
if not capture.isOpened():
    print("キャプチャーボードが開けませんでした")
    exit()

interval = 12  # 1分間に5回（60/5=12秒ごと）
window = 120    # 集計ウィンドウ（秒）
results = []   # 推論結果を保存
window_start = time.time()
last_pred_time = time.time()

# ---- POST 先の API ----
api_url = os.getenv("API_URL")  # 環境変数から取得
if not api_url:
    print("API_URL が設定されていません")
    exit()

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
        img_input = np.expand_dims(img_norm, axis=0)  # (1,128,128,3)

        pred = model.predict(img_input)
        class_id = int(np.argmax(pred))
        confidence = float(np.max(pred))

        results.append((class_id, confidence))
        print(f"推論: {class_names[class_id]}, 信頼度: {confidence}")

        last_pred_time = now

    # window秒ごとに集計してAPI送信
    if now - window_start >= window and results:
        # 最頻値のクラスIDを取得
        class_ids = [r[0] for r in results]
        most_common_id = max(set(class_ids), key=class_ids.count)
        # そのクラスの最大信頼度
        max_conf = max([r[1] for r in results if r[0] == most_common_id])

        result = {
            "class": class_names[most_common_id],
            "confidence": max_conf,
            "timestamp": datetime.datetime.now().isoformat()
        }
        print("API送信:", result)

        try:
            response = requests.post(api_url, json=result, timeout=10)
            if response.status_code == 200:
                print("API 送信成功:", response.json())
            else:
                print("API エラー:", response.status_code, response.text)
        except Exception as e:
            print("API 送信エラー:", e)

        # リストとwindow開始時刻をリセット
        results = []
        window_start = now

    cv2.imshow("Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()