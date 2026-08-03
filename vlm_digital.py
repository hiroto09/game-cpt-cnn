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
# Switch状態
# ======================================

last_packet = False

last_api_check = 0

CHECK_INTERVAL = 60


# ======================================
# 推論状態
# ======================================

# 現在推論を行っているか
inference_active = False

# 推論確定済みか
inference_confirmed = False

# 直前に推定されたID
last_prediction_id = None

# 同じIDが連続した回数
same_prediction_count = 0


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
        # ID
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

                confidence = (
                    line.split(
                        ":",
                        1
                    )[1].strip()
                )

            except IndexError:

                confidence = None


        # ==================================
        # 根拠
        # ==================================

        elif line.startswith(
            "根拠:"
        ):

            try:

                reason = (
                    line.split(
                        ":",
                        1
                    )[1].strip()
                )

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

        return True


    except requests.RequestException as e:

        print(
            "❌ 結果送信エラー"
        )

        print(e)

        return False


# ======================================
# 推論状態リセット
# ======================================

def reset_prediction_state():

    global inference_active
    global inference_confirmed
    global last_prediction_id
    global same_prediction_count

    inference_active = False
    inference_confirmed = False
    last_prediction_id = None
    same_prediction_count = 0

    print(
        "🔄 推論状態をリセットしました"
    )


# ======================================
# 推定結果処理
# ======================================

def process_prediction(
    result
):

    global inference_active
    global inference_confirmed
    global last_prediction_id
    global same_prediction_count


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


    if class_id is None:

        print(
            "⚠️ class_idを取得できませんでした"
        )

        return


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
    # 0の場合
    # ==================================
    #
    # 0は「ゲームなし」なので
    # 何回続いても推論を継続する
    #

    if class_id == 0:

        print(
            "⚪ id=0 のため推論を継続します"
        )

        # 0も結果として送信
        send_result(0)

        # 0は連続判定の対象外
        last_prediction_id = None
        same_prediction_count = 0

        inference_active = True
        inference_confirmed = False

        return


    # ==================================
    # 1～7の場合
    # ==================================

    # 初回のゲームID
    if last_prediction_id is None:

        last_prediction_id = class_id

        same_prediction_count = 1

        print(
            f"🎮 初回推定: id={class_id}"
        )

        # 初回結果も送信
        send_result(class_id)

        print(
            "🔍 もう一度同じIDが出るか確認します"
        )

        inference_active = True

        return


    # ==================================
    # 前回と同じID
    # ==================================

    if class_id == last_prediction_id:

        same_prediction_count += 1

        print(
            f"🎯 同じIDが連続 "
            f"{same_prediction_count}回"
        )


    # ==================================
    # 前回と違うID
    # ==================================

    else:

        print(
            f"🔄 ID変更: "
            f"{last_prediction_id} → {class_id}"
        )

        last_prediction_id = class_id

        same_prediction_count = 1

        # 違う結果も送信
        send_result(class_id)

        print(
            "🔍 新しいIDについて再確認します"
        )

        inference_active = True

        return


    # ==================================
    # 同じIDが2回連続
    # ==================================

    if same_prediction_count >= 2:

        print(
            f"🎯 id={class_id} を確定しました"
        )

        print(
            "🛑 Gemini推論を停止します"
        )

        inference_confirmed = True

        inference_active = False

    else:

        inference_active = True


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
                # Switch ON
                # ==================================

                if packet:

                    # ----------------------------------
                    # OFF → ON
                    # ----------------------------------

                    if not last_packet:

                        print(
                            "🎮 Switch起動検知"
                        )

                        print(
                            "🔍 新しいゲームの推定を開始します"
                        )

                        reset_prediction_state()

                        inference_active = True


                    # ----------------------------------
                    # 推論中
                    # ----------------------------------

                    if inference_active:

                        print(
                            "🤖 推論実行"
                        )

                        result = recognize_boardgame(
                            frame
                        )

                        if result is not None:

                            process_prediction(
                                result
                            )

                        else:

                            print(
                                "⚠️ Gemini推定失敗"
                            )


                    # ----------------------------------
                    # 確定後
                    # ----------------------------------

                    elif inference_confirmed:

                        print(
                            "🎯 ゲーム確定済み"
                        )

                        print(
                            "Gemini推論は停止しています"
                        )


                    # ----------------------------------
                    # 状態表示
                    # ----------------------------------

                    else:

                        print(
                            "🎮 Switch起動中"
                        )


                # ==================================
                # Switch OFF
                # ==================================

                else:

                    print(
                        "💤 Switch停止中"
                    )


                    # ----------------------------------
                    # ON → OFF
                    # ----------------------------------

                    if last_packet:

                        print(
                            "🔌 Switchの電源OFFを検知"
                        )

                        print(
                            "⚪ id=0を送信します"
                        )

                        send_result(0)

                        # 推論状態を完全リセット
                        reset_prediction_state()


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