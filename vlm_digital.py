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

            if result:

                f.write(
                    result.strip()
                )

            f.write(
                "\n"
            )

            f.write(
                "========================================\n\n"
            )

    except Exception as e:

        print(
            "⚠️ ログ保存エラー"
        )

        print(e)


# ======================================
# Switch状態
# ======================================

# 前回のSwitch状態
last_packet = False


# ======================================
# インターバル設定
# ======================================

# 通常時
# 推論やSwitch状態確認
NORMAL_INTERVAL = 120


# エラー時
ERROR_INTERVAL = 300


# 次回処理時刻
next_action_time = 0


# ======================================
# 推論状態
# ======================================

# 現在Gemini推論を行う状態か
inference_active = False


# ゲームが確定したか
inference_confirmed = False


# 直前の推定ID
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


    try:



        response = client.models.generate_content(

            model="gemini-flash-latest",

            contents=[
                image,
                PROMPT
            ]

        )


        if response.text:

            return response.text


        print(
            "⚠️ Geminiから応答がありません"
        )

        return None


    except Exception as e:

        print(
            "Geminiエラー"
        )

        print(e)


        error = str(e)


        # ==================================
        # 429
        # ==================================

        if "429" in error:

            print(
                "⚠️ Gemini APIのクォータ超過"
            )

            print(
                "5分後に再試行します"
            )


        # ==================================
        # 503
        # ==================================

        elif "503" in error:

            print(
                "⚠️ Gemini 503エラー"
            )

            print(
                "5分後に再試行します"
            )


        # ==================================
        # その他
        # ==================================

        else:

            print(
                "⚠️ Geminiでエラーが発生しました"
            )

            print(
                "5分後に再試行します"
            )


        return None


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



        response.raise_for_status()




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



# ======================================
# 推定結果処理
# ======================================

def process_prediction(result):

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
        f"id: {class_id}"
    )

    print(
        f"信頼度: {confidence}"
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
    # ID = 0
    # ==================================
    #
    # ゲームをしていない
    #
    # 0の場合は何回続いても
    # 推論を継続する
    # ==================================

    if class_id == 0:


        # 0も送信
        send_result(0)


        # 連続判定をリセット

        last_prediction_id = None

        same_prediction_count = 0


        # 推論継続

        inference_active = True

        inference_confirmed = False


        return True


    # ==================================
    # ID = 1～7
    # ==================================


    # ----------------------------------
    # 初回のゲームID
    # ----------------------------------

    if last_prediction_id is None:

        last_prediction_id = class_id

        same_prediction_count = 1


        # 初回結果はすぐ送信

        send_result(
            class_id
        )


        inference_active = True


        return True


    # ----------------------------------
    # 同じID
    # ----------------------------------

    if class_id == last_prediction_id:

        same_prediction_count += 1



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


    # ----------------------------------
    # 違うID
    # ----------------------------------

    print(
        f"🔄 ID変更: "
        f"{last_prediction_id} → {class_id}"
    )


    # 新しいIDを記録

    last_prediction_id = class_id

    same_prediction_count = 1


    # 新しいIDもすぐ送信

    send_result(
        class_id
    )


    print(
        "🔍 新しいIDについてもう一度確認します"
    )


    inference_active = True

    inference_confirmed = False


    return True


# ======================================
# 次回処理時刻設定
# ======================================

def set_next_normal_action():

    global next_action_time

    next_action_time = (
        time.time()
        + NORMAL_INTERVAL
    )

    print(
        "⏱️ 次回処理: 2分後"
    )


def set_next_error_action():

    global next_action_time

    next_action_time = (
        time.time()
        + ERROR_INTERVAL
    )

    print(
        "⏱️ 次回処理: 5分後"
    )


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
        # 次回処理時刻になるまで待機
        # ==================================

        if time.time() < next_action_time:

            time.sleep(0.01)

            continue


        # ==================================
        # Switch API確認
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


                    result = recognize_boardgame(
                        frame
                    )


                    # ----------------------------------
                    # Gemini成功
                    # ----------------------------------

                    if result is not None:

                        process_prediction(
                            result
                        )


                        # 通常2分後

                        set_next_normal_action()


                    # ----------------------------------
                    # Gemini失敗
                    # ----------------------------------

                    else:

                        print(
                            "⚠️ Gemini推定失敗"
                        )


                        # エラーなので5分

                        set_next_error_action()


                # ----------------------------------
                # 確定後
                # ----------------------------------

                elif inference_confirmed:

                    print(
                        "🎯 ゲーム確定済み"
                    )


                    print(
                        "🛑 Gemini推論は停止しています"
                    )


                    print(
                        "🔍 Switch電源状態のみ確認します"
                    )


                    # 確定後は2分後に
                    # Switch状態を確認

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
                        "🔌 Switchの電源OFFを検知"
                    )


                    print(
                        "⚪ id=0を送信します"
                    )


                    send_result(0)


                    # 推論状態完全リセット

                    reset_prediction_state()


                # OFF中は2分後に再確認

                set_next_normal_action()


            # ==================================
            # 現在のSwitch状態保存
            # ==================================

            last_packet = packet


        # ==================================
        # 通信エラー
        # ==================================

        except requests.RequestException as e:

            print(
                "❌ 通信エラー"
            )

            print(e)


            # 通信エラーは5分

            set_next_error_action()


        # ==================================
        # その他エラー
        # ==================================

        except Exception as e:

            print(
                "❌ エラー"
            )

            print(e)


            # その他のエラーも5分

            set_next_error_action()


# ======================================
# Ctrl+C
# ======================================

except KeyboardInterrupt:

    print(
        "\n終了します"
    )


# ======================================
# 終了処理
# ======================================

finally:

    if capture is not None:

        capture.release()


    cv2.destroyAllWindows()


    print(
        "カメラを解放しました"
    )