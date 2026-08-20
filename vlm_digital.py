import os
import time
import cv2
import requests
import threading
import json

from PIL import Image
from dotenv import load_dotenv
from google import genai


# ======================================
# 環境変数
# ======================================

load_dotenv()

SWITCH_API_URL = os.getenv("SWITCH_API_URL")
RESULT_API_URL = os.getenv("RESULT_API_URL")
EVENTS_API_URL = os.getenv("EVENTS_API_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not SWITCH_API_URL:
    raise ValueError("SWITCH_API_URL が設定されていません")

if not RESULT_API_URL:
    raise ValueError("RESULT_API_URL が設定されていません")

if not EVENTS_API_URL:
    raise ValueError("EVENTS_API_URL が設定されていません")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY が設定されていません")


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


# ======================================
# ゲーム候補取得
# ======================================

def get_game_candidates():
    """
    StayWatch APIからdigitalゲーム一覧を取得する。
    """

    try:
        response = requests.get(
            EVENTS_API_URL,
            params={
                "game_type": "digital"
            },
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        games = data["data"]

        candidates = "\n".join(
            f'    "{game["ID"]}" : "{game["Name"]}"'
            for game in games
        )

        print("🎮 digitalゲーム候補をAPIから取得しました")

        for game in games:
            print(
                f'    ID: {game["ID"]}, '
                f'Name: {game["Name"]}'
            )

        return candidates

    except (
        requests.RequestException,
        KeyError,
        TypeError,
        ValueError
    ) as e:

        print("❌ digitalゲーム候補の取得に失敗しました")
        print(e)

        return None


# ======================================
# Prompt作成
# ======================================

def create_prompt():
    """
    prompt.txtを読み込み、
    {GAME_CANDIDATES} をAPIから取得した候補一覧に置き換える。
    """

    with open(
        PROMPT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        prompt_template = f.read()

    candidates = get_game_candidates()

    if candidates is None:
        raise RuntimeError(
            "digitalゲーム候補を取得できないため、"
            "Gemini推論を実行できません"
        )

    prompt = prompt_template.replace(
        "{GAME_CANDIDATES}",
        candidates
    )

    return prompt


print("✅ prompt.txt を読み込みました")


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

            f.write("\n")

            f.write(
                "========================================\n\n"
            )

        print(
            f"📝 ログ保存: {LOG_FILE}"
        )

    except Exception as e:

        print("⚠️ ログ保存エラー")
        print(e)


# ======================================
# インターバル
# ======================================

# 通常時：3分
NORMAL_INTERVAL = 180

# ゲーム確定後：30分
CONFIRMED_INTERVAL = 1800

# エラー時：5分
ERROR_INTERVAL = 300

# Switch状態確認間隔
SWITCH_CHECK_INTERVAL = 5


# ======================================
# Switch状態
# ======================================

last_packet = False


# ======================================
# 推論状態
# ======================================

# Gemini推論を実行可能か
inference_active = False

# ゲームが確定して30分待機中か
inference_confirmed = False

# 前回の推定ID
last_prediction_id = None

# 同じIDの連続回数
same_prediction_count = 0

# 次回推論可能時刻
next_action_time = 0


# ======================================
# Gemini状態
# ======================================

gemini_running = False

gemini_result = None

gemini_error = False

gemini_thread = None

# スレッド間のデータ保護
gemini_lock = threading.Lock()


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

    print("⚠️ カメラ再接続")

    capture.release()

    time.sleep(2)

    capture = open_camera()

    if capture is None:

        raise RuntimeError(
            "❌ カメラ再接続失敗"
        )

    ret, frame = capture.read()

    if not ret:

        raise RuntimeError(
            "❌ 画像取得失敗"
        )

    return frame


# ======================================
# Switch状態取得
# ======================================

def get_switch_state():

    try:

        response = requests.get(
            SWITCH_API_URL,
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        if "packet" not in data:

            raise ValueError(
                "Switch APIレスポンスにpacketがありません"
            )

        return bool(data["packet"])

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError
    ) as e:

        print("❌ Switch状態取得エラー")
        print(e)

        return None


# ======================================
# 結果送信
# ======================================

def send_result(class_id):

    try:

        print("================================")
        print("📤 結果送信")
        print(f"class_id: {class_id}")

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

        return True

    except requests.RequestException as e:

        print("❌ 結果送信エラー")
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
    global next_action_time

    inference_active = False
    inference_confirmed = False

    last_prediction_id = None
    same_prediction_count = 0

    next_action_time = 0

    print("🔄 推論状態をリセットしました")


# ======================================
# Gemini推論本体
# ======================================

def recognize_boardgame(frame):

    global gemini_result
    global gemini_error

    with gemini_lock:

        gemini_result = None
        gemini_error = False

    try:

        print("🤖 Gemini API送信開始")

        # 推論直前に最新のゲーム候補を取得
        prompt = create_prompt()

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
                prompt
            ]
        )

        print("🤖 Gemini API応答受信")

        if not response.text:

            print(
                "⚠️ Geminiから回答がありません"
            )

            with gemini_lock:
                gemini_error = True

            return

        with gemini_lock:
            gemini_result = response.text

    except Exception as e:

        print("❌ Geminiエラー")
        print(e)

        with gemini_lock:
            gemini_error = True


# ======================================
# Gemini推論開始
# ======================================

def start_gemini_inference(frame):

    global gemini_running
    global gemini_thread

    with gemini_lock:

        if gemini_running:

            return False

        gemini_running = True

    print("🔍 ゲーム推定開始")

    def worker():

        global gemini_running

        try:

            recognize_boardgame(frame)

        finally:

            with gemini_lock:
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

    with gemini_lock:

        if gemini_running:

            return None, False

        if gemini_error:

            gemini_error = False

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

    # ----------------------------------
    # まずJSONとして解析
    # ----------------------------------

    try:

        text = result.strip()

        # ```json ... ``` を除去
        if text.startswith("```"):

            lines = text.splitlines()

            if len(lines) >= 3:

                text = "\n".join(
                    lines[1:-1]
                )

        data = json.loads(text)

        class_id = data.get("id")

        confidence = data.get(
            "confidence"
        )

        reason = data.get(
            "reason"
        )

        if class_id is not None:

            class_id = int(class_id)

        return (
            class_id,
            confidence,
            reason
        )

    except (
        json.JSONDecodeError,
        ValueError,
        TypeError
    ):

        pass


    # ----------------------------------
    # JSON解析失敗時は従来形式を解析
    # ----------------------------------

    for line in result.splitlines():

        line = line.strip()

        lower_line = line.lower()

        # ID
        if (
            lower_line.startswith("id:")
            or lower_line.startswith("id：")
        ):

            try:

                value = (
                    line.split(
                        ":",
                        1
                    )[1]
                    if ":" in line
                    else line.split(
                        "：",
                        1
                    )[1]
                )

                class_id = int(
                    value.strip()
                )

            except (
                ValueError,
                IndexError
            ):

                print(
                    "⚠️ ID解析失敗:",
                    line
                )

        # 信頼度
        elif (
            line.startswith("信頼度:")
            or line.startswith("信頼度：")
        ):

            try:

                confidence = (
                    line.split(
                        ":",
                        1
                    )[1]
                    if ":" in line
                    else line.split(
                        "：",
                        1
                    )[1]
                ).strip()

            except IndexError:

                confidence = None

        # 根拠
        elif (
            line.startswith("根拠:")
            or line.startswith("根拠：")
        ):

            try:

                reason = (
                    line.split(
                        ":",
                        1
                    )[1]
                    if ":" in line
                    else line.split(
                        "：",
                        1
                    )[1]
                ).strip()

            except IndexError:

                reason = None

    return (
        class_id,
        confidence,
        reason
    )


# ======================================
# 次回処理時刻
# ======================================

def set_next_normal_action():

    global next_action_time

    next_action_time = (
        time.time()
        + NORMAL_INTERVAL
    )


def set_next_error_action():

    global next_action_time

    next_action_time = (
        time.time()
        + ERROR_INTERVAL
    )


# ======================================
# 推定結果処理
# ======================================

def process_prediction(result):

    global inference_active
    global inference_confirmed
    global last_prediction_id
    global same_prediction_count
    global next_action_time

    (
        class_id,
        confidence,
        reason
    ) = parse_result(result)

    # ----------------------------------
    # ID取得失敗
    # ----------------------------------

    if class_id is None:

        print(
            "⚠️ class_idを取得できませんでした"
        )

        return False

    # ----------------------------------
    # 結果表示
    # ----------------------------------

    print("================================")
    print("🎮 推定結果")
    print(f"id: {class_id}")
    print(f"信頼度: {confidence}")
    print(f"根拠: {reason}")
    print("================================")

    # ----------------------------------
    # ログ
    # ----------------------------------

    write_prediction_log(
        result=result,
        class_id=class_id,
        confidence=confidence,
        reason=reason
    )

    # ----------------------------------
    # ID = 0
    # ----------------------------------

    if class_id == 0:

        if send_result(0):

            last_prediction_id = None
            same_prediction_count = 0

            # 0の場合も3分後に再確認
            inference_active = False
            inference_confirmed = False

            set_next_normal_action()

            return True

        print("❌ id=0送信失敗")

        return False

    # ----------------------------------
    # 初回ゲームID
    # ----------------------------------

    if last_prediction_id is None:

        print(
            f"🎮 初回推定: id={class_id}"
        )

        if send_result(class_id):

            last_prediction_id = class_id
            same_prediction_count = 1

            # 次回推論可能
            inference_active = False

            set_next_normal_action()

            return True

        print("❌ 結果送信失敗")

        return False

    # ----------------------------------
    # 同じID
    # ----------------------------------

    if class_id == last_prediction_id:

        same_prediction_count += 1

        print(
            f"🔁 同じID: "
            f"{same_prediction_count}回連続"
        )

        # ----------------------------------
        # 2回連続 → ゲーム確定
        # ----------------------------------

        if same_prediction_count >= 2:

            inference_confirmed = True
            inference_active = False

            next_action_time = (
                time.time()
                + CONFIRMED_INTERVAL
            )

            print(
                "✅ ゲームを確定しました"
            )

            print(
                "⏰ 30分間Switch状態を監視します"
            )

            return True

        inference_active = False

        set_next_normal_action()

        return True

    # ----------------------------------
    # 違うID
    # ----------------------------------

    print(
        f"🔄 ID変更: "
        f"{last_prediction_id} → {class_id}"
    )

    if send_result(class_id):

        last_prediction_id = class_id
        same_prediction_count = 1

        inference_active = False

        set_next_normal_action()

        return True

    print("❌ 結果送信失敗")

    return False


# ======================================
# メイン
# ======================================

try:

    while True:

        # ==================================
        # カメラ画像取得
        # ==================================

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
        # Gemini推論中
        # ==================================

        if gemini_running:

            time.sleep(0.01)

            continue

        # ==================================
        # Gemini結果確認
        # ==================================

        result, error = get_gemini_result()

        if result is not None:

            success = process_prediction(
                result
            )

            if not success:

                set_next_error_action()

            continue

        if error:

            print(
                "⚠️ Gemini推論失敗"
            )

            set_next_error_action()

            continue

        # ==================================
        # Switch状態取得
        # ==================================

        packet = get_switch_state()

        # ==================================
        # API通信エラー
        # ==================================

        if packet is None:

            set_next_error_action()

            time.sleep(
                ERROR_INTERVAL
            )

            continue

        # ==================================
        # Switch OFF
        # ==================================

        if not packet:

            # ON → OFF
            if last_packet:

                print(
                    "🔌 Switch電源OFFを検知"
                )

                print(
                    "⚪ id=0を送信します"
                )

                send_result(0)

                reset_prediction_state()

            else:

                print(
                    "⚪ Switch停止中"
                )

            last_packet = False

            time.sleep(
                SWITCH_CHECK_INTERVAL
            )

            continue

        # ==================================
        # Switch ON
        # ==================================

        print(
            "🟢 Switch起動中"
        )

        # ==================================
        # OFF → ON
        # ==================================

        if not last_packet:

            print(
                "🔄 OFF → ONを検知"
            )

            reset_prediction_state()

            inference_active = True

        # ==================================
        # 30分確定中
        # ==================================

        if inference_confirmed:

            remaining = (
                next_action_time
                - time.time()
            )

            if remaining > 0:

                print(
                    f"🔒 ゲーム確定中 "
                    f"(残り約{int(remaining)}秒)"
                )

            else:

                print(
                    "⏰ 30分経過"
                )

                print(
                    "🔍 再度ゲーム推論開始"
                )

                inference_confirmed = False
                inference_active = True

                last_prediction_id = None
                same_prediction_count = 0

        # ==================================
        # 通常推論
        # ==================================

        elif inference_active:

            if (
                time.time()
                >= next_action_time
            ):

                started = (
                    start_gemini_inference(
                        frame.copy()
                    )
                )

                if not started:

                    print(
                        "⚠️ Gemini推論開始できません"
                    )

        # ==================================
        # 次回推論待機中
        # ==================================

        else:

            if time.time() < next_action_time:

                remaining = (
                    next_action_time
                    - time.time()
                )

                print(
                    f"⏳ 次回推論まで "
                    f"約{int(remaining)}秒"
                )

        # ==================================
        # 現在状態保存
        # ==================================

        last_packet = packet

        # ==================================
        # 次のSwitch確認
        # ==================================

        time.sleep(
            SWITCH_CHECK_INTERVAL
        )


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