import os
import time
import cv2
import requests

from PIL import Image
from dotenv import load_dotenv
from google import genai

# ======================================
# 環境変数
# ======================================
load_dotenv()

SWITCH_API_URL = os.getenv("SWITCH_API_URL")
RESULT_API_URL = os.getenv("RESULT_API_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SWITCH_API_URL:
    raise ValueError("SWITCH_API_URL が設定されていません")

if not RESULT_API_URL:
    raise ValueError("RESULT_API_URL が設定されていません")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY が設定されていません")

# ======================================
# Gemini
# ======================================
client = genai.Client(api_key=GEMINI_API_KEY)

# ======================================
# Prompt
# ======================================
with open("prompt.txt", "r", encoding="utf-8") as f:
    PROMPT = f.read()

# ======================================
# 前回状態
# ======================================
last_packet = False
last_api_check = 0
CHECK_INTERVAL = 60
# ======================================
# カメラ
# ======================================
def open_camera():

    for i in range(3):

        cap = cv2.VideoCapture(i)

        if cap.isOpened():
            print(f"✅ カメラ{i}に接続")
            return cap

        cap.release()

    return None


capture = open_camera()
cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Preview", 960, 540)

if capture is None:
    raise RuntimeError("❌ カメラが見つかりません")

# ======================================
# 画像取得
# ======================================
def capture_image():

    global capture

    ret, frame = capture.read()

    if ret:
        return frame

    print("⚠️ カメラ再接続")

    capture.release()

    time.sleep(2)

    capture = open_camera()

    if capture is None:
        raise RuntimeError("カメラ再接続失敗")

    ret, frame = capture.read()

    if not ret:
        raise RuntimeError("画像取得失敗")

    return frame

# ======================================
# Gemini推論
# ======================================
def recognize_boardgame(frame):

    image = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    while True:

        try:

            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[
                    image,
                    PROMPT
                ]
            )

            if response.text:
                return response.text

            raise RuntimeError("Geminiから応答がありません")

        except Exception as e:

            print("Geminiエラー")
            print(e)

            if "503" in str(e):
                print("503エラーのため30秒待機")
                time.sleep(30)
                continue

            print("10秒後に再試行")
            time.sleep(10)

# ======================================
# 結果送信
# ======================================
def send_result(result):

    response = requests.post(
        RESULT_API_URL,
        json={
            "result": result
        },
        timeout=10
    )

    response.raise_for_status()

    print("✅ 結果送信完了")

# ======================================
# メイン
# ======================================
try:

    while True:

        # 映像取得
        ret, frame = capture.read()

        if not ret:
            frame = capture_image()

        # プレビュー表示
        cv2.imshow("Preview", frame)

        # qで終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # 60秒ごとにAPI確認
        if time.time() - last_api_check >= CHECK_INTERVAL:

            last_api_check = time.time()

            try:

                response = requests.get(
                    SWITCH_API_URL,
                    timeout=5
                )

                response.raise_for_status()

                packet = response.json()["packet"]

                if packet and not last_packet:

                    print("🎮 Switch起動検知")

                    print("🤖 推論開始")

                    result = recognize_boardgame(frame)

                    print(result)

                    send_result(result)

                elif packet:

                    print("🎮 Switch起動中（監視のみ）")

                else:

                    print("💤 Switch停止中")

                last_packet = packet

            except requests.RequestException as e:

                print("通信エラー")
                print(e)

            except Exception as e:

                print("エラー")
                print(e)

        # CPU負荷軽減
        time.sleep(0.01)

except KeyboardInterrupt:

    print("\n終了します")

finally:

    capture.release()
    cv2.destroyAllWindows()

    print("カメラを解放しました")

