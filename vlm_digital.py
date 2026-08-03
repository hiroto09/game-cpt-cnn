import os
import time
import cv2
import requests
import threading

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

PROMPT_FILE = "prompt.txt"


if not os.path.exists(PROMPT_FILE):

    raise FileNotFoundError(
        f"{PROMPT_FILE} が見つかりません"
    )


with open(
    PROMPT_FILE,
    "r",
    encoding="utf-8"
) as f:

    PROMPT = f.read()


print(
    "✅ prompt.txt を読み込みました"
)


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


            if result:

                f.write(
                    result.strip()
                )

            else:

                f.write(
                    "Geminiから回答なし"
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


# ======================================
# インターバル
# ======================================

# 通常時
NORMAL_INTERVAL = 120


# エラー時
ERROR_INTERVAL = 300


# 次回Switch確認時刻
next_action_time = 0


# ======================================
# 推論状態
# ======================================

# 推論を実行する状態か
inference_active = False


# 推論確定済みか
inference_confirmed = False


# 前回の推定ID
last_prediction_id = None


# 同じIDの連続回数
same_prediction_count = 0


# ======================================
# Geminiスレッド状態
# ======================================

# 現在Gemini推論中か
gemini_running = False


# Gemini推論結果
gemini_result = None


# Geminiエラーが発生したか
gemini_error = False


# Geminiスレッド
gemini_thread = None


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
# プレビュー
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
# カメラ画像取得
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
# Gemini推論本体
# ======================================

def recognize_boardgame(frame):

    global gemini_result
    global gemini_error


    gemini_result = None
    gemini_error = False


    try:

        print(
            "🤖 Gemini API送信開始"
        )


        image = Image.fromarray(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
        )


        response = client.models.generate_content(

            model="gemini-flash-latest",

            contents=[
                image,
                PROMPT
            ]

        )


        print(
            "🤖 Gemini API応答受信"
        )


        if response.text:

            gemini_result = response.text

            print(
                "✅ Gemini推論成功"
            )

        else:

            print(
                "⚠️ Geminiから応答がありません"
            )

            gemini_error = True


    except Exception as e:

        print(
            "❌ Geminiエラー"
        )

        print(e)


        gemini_error = True


# ======================================
# Gemini推論開始
# ======================================

def start_gemini_inference(frame):

    global gemini_running
    global gemini_thread


    if gemini_running:

        print(
            "⚠️ Gemini推論はすでに実行中です"
        )

        return False


    gemini_running = True


    def worker():

        global gemini_running


        try:

            recognize_boardgame(
                frame
            )

        finally:

            gemini_running = False


    gemini_thread = threading.Thread(
        target=worker,
        daemon=True
    )


    gemini_thread.start()


    return True


# ======================================
# Gemini結果取得
# ======================================

def get_gemini_result():

    global gemini_result
    global gemini_error


    if gemini_running:

        return None, False


    if gemini_error:

        return None, True


    if gemini_result is not None:

        result = gemini_result

        gemini_result = None

        return result, False


    return None, False


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

        if line.lower().startswith(
            "id:"
        ):

            try:

                value = line.split(
                    ":",
                    1
                )[1].strip()


                class_id = int(
                    value
                )


            except (
                ValueError,
                IndexError
            ):

                print(
                    "⚠️ ID解析失敗:",
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
            f"class_id: {class_id}"
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

def process_prediction(result):

    global inference_active
    global inference_confirmed
    global last_prediction_id
    global same_prediction_count


    (
        class_id,
        confidence,
        reason
    ) = parse_result(
        result
    )


    # ==================================
    # ID取得失敗
    # ==================================

    if class_id is None:

        print(
            "⚠️ class_idを取得できませんでした"
        )

        return False


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
    # ログ
    # ==================================

    write_prediction_log(

        result=result,

        class_id=class_id,

        confidence=confidence,

        reason=reason

    )


    # ==================================
    # ID = 0
    # ==================================

    if class_id == 0:

        print(
            "⚪ id=0"
        )

        print(
            "🔍 推論を継続します"
        )


        send_result(0)


        last_prediction_id = None

        same_prediction_count = 0


        inference_active = True

        inference_confirmed = False


        return True


    # ==================================
    # 初回のゲームID
    # ==================================

    if last_prediction_id is None:

        last_prediction_id = class_id

        same_prediction_count = 1


        print(
            f"🎮 初回推定: id={class_id}"
        )


        # 初回結果は即送信
        send_result(
            class_id
        )


        inference_active = True


        print(
            "🔍 もう一度同じIDが出るか確認します"
        )


        return True


    # ==================================
    # 同じID
    # ==================================

    if class_id == last_prediction_id:

        same_prediction_count += 1


        print(
            f"🎯 同じIDが連続 "
            f"{same_prediction_count}回"
        )


        # ==================================
        # 2回連続
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


            return True


        inference_active = True

        return True


    # ==================================
    # 違うID
    # ==================================

    print(
        f"🔄 ID変更: "
        f"{last_prediction_id} → {class_id}"
    )


    last_prediction_id = class_id

    same_prediction_count = 1


    # 違うIDも即送信
    send_result(
        class_id
    )


    print(
        "🔍 新しいIDについて再確認します"
    )


    inference_active = True

    inference_confirmed = False


    return True


# ======================================
# 次回処理時刻
# ======================================

def set_next_normal_action():

    global next_action_time


    next_action_time = (
        time.time()
        + NORMAL_INTERVAL
    )


    print(
        "⏱️ 次回確認: 2分後"
    )


def set_next_error_action():

    global next_action_time


    next_action_time = (
        time.time()
        + ERROR_INTERVAL
    )


    print(
        "⏱️ 次回確認: 5分後"
    )


# ======================================
# メイン
# ======================================

try:

    while True:

        # ==================================
        # カメラ
        # ==================================

        ret, frame = capture.read()


        if not ret:

            frame = capture_image()


        # ==================================
        # プレビュー
        # ==================================

        cv2.imshow(
            "Preview",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


        # ==================================
        # Gemini結果確認
        # ==================================

        if gemini_running:

            # 推論中はプレビューだけ動かす

            time.sleep(
                0.01
            )

            continue


        # ==================================
        # Gemini結果が返ってきたか確認
        # ==================================

        if gemini_result is not None:

            result, error = get_gemini_result()


            if result is not None:

                process_prediction(
                    result
                )


                # 成功したので2分後

                set_next_normal_action()


                continue


            if error:

                print(
                    "⚠️ Gemini推論失敗"
                )


                set_next_error_action()


                continue


        # ==================================
        # 次回処理時刻
        # ==================================

        if time.time() < next_action_time:

            time.sleep(
                0.01
            )

            continue


        # ==================================
        # Switch状態確認
        # ==================================

        try:

            print(
                "🔍 Switch状態確認"
            )


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


                    # Geminiを別スレッドで開始

                    started = start_gemini_inference(
                        frame.copy()
                    )


                    if started:

                        # 推論成功/失敗は
                        # Gemini終了後に判定

                        pass


                # ----------------------------------
                # 確定後
                # ----------------------------------

                elif inference_confirmed:

                    print(
                        "🎯 ゲーム確定済み"
                    )


                    print(
                        "🛑 Gemini推論は停止中"
                    )


                    print(
                        "🔍 Switch状態のみ監視"
                    )


                    set_next_normal_action()


                else:

                    print(
                        "🎮 Switch起動中"
                    )


                    set_next_normal_action()


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
                        "🔌 Switch電源OFFを検知"
                    )


                    print(
                        "⚪ id=0を送信します"
                    )


                    send_result(0)


                    reset_prediction_state()


                set_next_normal_action()


            # ==================================
            # 現在状態保存
            # ==================================

            last_packet = packet


        # ==================================
        # 通信エラー
        # ==================================

        except requests.RequestException as e:

            print(
                "❌ Switch通信エラー"
            )

            print(e)


            set_next_error_action()


        # ==================================
        # その他エラー
        # ==================================

        except Exception as e:

            print(
                "❌ エラー"
            )

            print(e)


            set_next_error_action()


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