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
    raise ValueError(
        "SWITCH_API_URL が設定されていません"
    )

if not RESULT_API_URL:
    raise ValueError(
        "RESULT_API_URL が設定されていません"
    )

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY が設定されていません"
    )


# ======================================
# Gemini
# ======================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ======================================
# Prompt
# ======================================

with open(
    "prompt.txt",
    "r",
    encoding="utf-8"
) as f:

    PROMPT = f.read()


# ======================================
# ログ設定
# ======================================

LOG_DIR = "logs"

LOG_FILE = os.path.join(
    LOG_DIR,
    "game_prediction.log"
)


# logsディレクトリがなければ作成
os.makedirs(
    LOG_DIR,
    exist_ok=True
)


# ======================================
# ログ書き込み
# ======================================

def write_prediction_log(
    result,
    class_id,
    confidence,
    reason
):

    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    try:

        with open(
            LOG_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                "========================================\n"
            )

            f.write(
                f"日時: {timestamp}\n"
            )

            f.write(
                f"id: {class_id}\n"
            )

            f.write(
                f"信頼度: {confidence}\n"
            )

            f.write(
                f"根拠: {reason}\n"
            )

            f.write(
                "Gemini生回答:\n"
            )

            f.write(
                result.strip()
            )

            f.write(
                "\n"
            )

            f.write(
                "========================================\n\n"
            )

        print(
            f"📝 ログ保存: {LOG_FILE}"
        )

    except Exception as e:

        print(
            "⚠️ ログ保存エラー"
        )

        print(e)


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

            print(
                f"✅ カメラ{i}に接続"
            )

            return cap

        cap.release()

    return None


capture = open_camera()


if capture is None:

    raise RuntimeError(
        "❌ カメラが見つかりません"
    )


# ======================================
# プレビューウィンドウ
# ======================================

cv2.namedWindow(
    "Preview",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Preview",
    960,
    540
)


# ======================================
# 画像取得
# ======================================

def capture_image():

    global capture

    ret, frame = capture.read()

    if ret:

        return frame

    print(
        "⚠️ カメラ再接続"
    )

    capture.release()

    time.sleep(2)

    capture = open_camera()

    if capture is None:

        raise RuntimeError(
            "カメラ再接続失敗"
        )

    ret, frame = capture.read()

    if not ret:

        raise RuntimeError(
            "画像取得失敗"
        )

    return frame


# ======================================
# Gemini推論
# ======================================

def recognize_boardgame(frame):

    image = Image.fromarray(
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
    )

    while True:

        try:

            print(
                "🤖 Gemini推論開始"
            )

            response = client.models.generate_content(

                model="gemini-flash-latest",

                contents=[
                    image,
                    PROMPT
                ]

            )

            if response.text:

                return response.text

            raise RuntimeError(
                "Geminiから応答がありません"
            )


        except Exception as e:

            print(
                "Geminiエラー"
            )

            print(e)

            error = str(e)


            # ==================================
            # 503
            # ==================================

            if "503" in error:

                print(
                    "503エラーのため60秒待機"
                )

                time.sleep(60)

                continue


            # ==================================
            # 429
            # ==================================

            if "429" in error:

                print(
                    "⚠️ Gemini APIのクォータ超過"
                )

                print(
                    "今回の推論を中止します"
                )

                return None


            # ==================================
            # その他
            # ==================================

            print(
                "60秒後に再試行"
            )

            time.sleep(60)


# ======================================
# Gemini結果解析
# ======================================

def parse_result(result):

    if not result:

        return None, None, None


    class_id = None

    confidence = None

    reason = None


    for line in result.splitlines():

        line = line.strip()


        # ==================================
        # id
        # ==================================

        if line.lower().startswith("id:"):

            try:

                value = line.split(
                    ":",
                    1
                )[1].strip()

                class_id = int(value)

            except (
                ValueError,
                IndexError
            ):

                print(
                    "⚠️ IDの解析に失敗:",
                    line
                )


        # ==================================
        # 信頼度
        # ==================================

        elif line.startswith(
            "信頼度:"
        ):

            try:

                value = line.split(
                    ":",
                    1
                )[1].strip()

                confidence = value

            except IndexError:

                confidence = None


        # ==================================
        # 根拠
        # ==================================

        elif line.startswith(
            "根拠:"
        ):

            try:

                reason = line.split(
                    ":",
                    1
                )[1].strip()

            except IndexError:

                reason = None


    return (
        class_id,
        confidence,
        reason
    )


# ======================================
# 結果送信
# ======================================

def send_result(class_id):

    try:

        print(
            "================================"
        )

        print(
            "📤 結果送信"
        )

        print(
            "URL:",
            RESULT_API_URL
        )

        print(
            "class_id:",
            class_id
        )


        response = requests.post(

            RESULT_API_URL,

            json={
                "class_id": class_id
            },

            timeout=10

        )


        print(
            "HTTP Status:",
            response.status_code
        )

        print(
            "Response:",
            response.text
        )


        response.raise_for_status()


        print(
            "✅ 結果送信完了"
        )


    except requests.RequestException as e:

        print(
            "❌ 結果送信エラー"
        )

        print(e)


# ======================================
# メイン
# ======================================

try:

    while True:


        # ==================================
        # カメラ映像取得
        # ==================================

        ret, frame = capture.read()

        if not ret:

            frame = capture_image()


        # ==================================
        # プレビュー表示
        # ==================================

        cv2.imshow(
            "Preview",
            frame
        )


        # ==================================
        # qで終了
        # ==================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


        # ==================================
        # 60秒ごとにSwitch API確認
        # ==================================

        if (
            time.time()
            - last_api_check
            >= CHECK_INTERVAL
        ):

            last_api_check = time.time()


            try:

                response = requests.get(

                    SWITCH_API_URL,

                    timeout=5

                )

                response.raise_for_status()


                packet = response.json()["packet"]


                # ==================================
                # False → True
                # ==================================

                if packet and not last_packet:

                    print(
                        "🎮 Switch起動検知"
                    )


                    # ==================================
                    # Gemini推論
                    # ==================================

                    result = recognize_boardgame(
                        frame
                    )


                    # ==================================
                    # 推定失敗
                    # ==================================

                    if result is None:

                        print(
                            "⚠️ Gemini推定失敗"
                        )

                    else:

                        # ==================================
                        # 結果解析
                        # ==================================

                        (
                            class_id,
                            confidence,
                            reason
                        ) = parse_result(
                            result
                        )


                        # ==================================
                        # 結果表示
                        # ==================================

                        print(
                            "================================"
                        )

                        print(
                            "🎮 推定結果"
                        )

                        print(
                            f"id: {class_id}"
                        )

                        print(
                            f"信頼度: {confidence}"
                        )

                        print(
                            f"根拠: {reason}"
                        )

                        print(
                            "================================"
                        )


                        # ==================================
                        # ログ保存
                        # ==================================

                        write_prediction_log(

                            result=result,

                            class_id=class_id,

                            confidence=confidence,

                            reason=reason

                        )


                        # ==================================
                        # class_idが取得できた場合
                        # ==================================

                        if class_id is not None:

                            send_result(
                                class_id
                            )

                        else:

                            print(
                                "⚠️ class_idを取得できなかったため送信しません"
                            )


                elif packet:

                    print(
                        "🎮 Switch起動中（監視のみ）"
                    )


                else:

                    print(
                        "💤 Switch停止中"
                    )


                # ==================================
                # 現在状態を保存
                # ==================================

                last_packet = packet


            except requests.RequestException as e:

                print(
                    "通信エラー"
                )

                print(e)


            except Exception as e:

                print(
                    "エラー"
                )

                print(e)


        # ==================================
        # CPU負荷軽減
        # ==================================

        time.sleep(0.01)


except KeyboardInterrupt:

    print(
        "\n終了します"
    )

finally:

    if capture is not None:

        capture.release()

    cv2.destroyAllWindows()

    print(
        "カメラを解放しました"
    )