import os
import time
import cv2
import requests
import threading
import json
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv
from google import genai

# ======================================
# 定数・設定
# ======================================
PROMPT_FILE = "prompt.txt"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "game_prediction.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "gemini_errors.log")

# インターバル設定（秒）
NORMAL_INTERVAL = 180         # 通常時：3分
CONFIRMED_INTERVAL = 3600     # ゲーム確定後：60分
ERROR_INTERVAL = 600          # 500エラー等：10分ストップ
RATE_LIMIT_INTERVAL = 3600    # 429エラー(Too Many Requests)時：1時間ストップ
SWITCH_CHECK_INTERVAL = 60    # Switch状態確認間隔：60秒

# フォールバック用のモデルリスト
FALLBACK_MODELS = [
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
]


# ======================================
# エラーログ書き込み関数
# ======================================
def write_error_log(error_type, target_or_model, detail=""):
    """
    429 / 503 等のエラー発生時に日時・エラー種別・対象モデル等をログファイルへ追記
    出力例: [2026-09-23 19:15:00] ERROR: 429 (Rate Limit) | TARGET/MODEL: gemini-flash-latest | DETAIL: ...
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] ERROR: {error_type} | TARGET/MODEL: {target_or_model}"
    if detail:
        log_line += f" | DETAIL: {detail}"
    log_line += "\n"

    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception as e:
        print("❌ エラーログ保存失敗:", e)


class GameRecognizerApp:
    def __init__(self):
        # 1. 環境変数の読み込みと検証
        load_dotenv()
        self.SWITCH_API_URL = self._get_env("SWITCH_API_URL")
        self.RESULT_API_URL = self._get_env("RESULT_API_URL")
        self.EVENTS_API_URL = self._get_env("EVENTS_API_URL")
        self.STAYWATCH_API_USERS = os.getenv("STAYWATCH_API_USERS")
        self.STAYWATCH_API_KEY = os.getenv("STAYWATCH_API_KEY")
        self.GEMINI_API_KEY = self._get_env("GEMINI_API_KEY")

        # 2. 初期化
        self.client = genai.Client(api_key=self.GEMINI_API_KEY)
        self.game_candidates_text = None
        self.member_candidates_text = None
        self.prompt = self._create_prompt()
        self.capture = self._open_camera()

        os.makedirs(LOG_DIR, exist_ok=True)
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview", 960, 540)

        # 3. 状態管理変数
        self.last_packet = False
        self.inference_active = False
        self.inference_confirmed = False
        self.last_prediction_id = None
        self.last_member_ids = []        # 複数の参加者IDリストを保持
        self.same_prediction_count = 0
        self.next_action_time = 0
        self.last_switch_check_time = 0  # Switch状態確認用のタイマー

        # Geminiスレッド用
        self.gemini_running = False
        self.gemini_result = None
        self.gemini_frame = None       # 推論に使用した画像フレーム
        self.gemini_error_code = None  # エラー発生時のステータスコードを保持
        self.gemini_lock = threading.Lock()

    def _get_env(self, key):
        """環境変数を取得し、存在しない場合はエラーを出す"""
        val = os.getenv(key)
        if not val:
            raise ValueError(f"{key} が設定されていません")
        return val

    # ======================================
    # 初期化・候補取得関連
    # ======================================
    def _get_game_candidates(self):
        """APIからdigitalゲーム一覧を取得して文字列として返す"""
        try:
            response = requests.get(self.EVENTS_API_URL, params={"game_type": "digital"}, timeout=10)
            if response.status_code in [429, 503]:
                write_error_log(f"{response.status_code} HTTP Error", "EVENTS_API_URL", response.text[:100])
            response.raise_for_status()
            games = response.json()["data"]

            print("🎮 digitalゲーム候補をAPIから取得しました")
            return "\n".join(f'    "{game["ID"]}" : "{game["Name"]}"' for game in games)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "503" in err_str:
                err_type = "429 Rate Limit" if "429" in err_str else "503 Service Unavailable"
                write_error_log(err_type, "EVENTS_API_URL", err_str[:100])
            print(f"❌ digitalゲーム候補の取得に失敗しました: {e}")
            return None

    def _get_member_candidates(self):
        """
        STAYWATCH_API_USERS から参加者一覧を取得し、
        「id<TAB>name」形式文字列を返す（取得できない場合は空文字を返して継続）
        """
        if not self.STAYWATCH_API_USERS or not self.STAYWATCH_API_KEY:
            print("⚠️ STAYWATCH_API_USERS または STAYWATCH_API_KEY が未設定のため、参加者候補なしで続行します")
            return ""

        headers = {
            "X-API-Key": self.STAYWATCH_API_KEY,
        }
        try:
            response = requests.get(self.STAYWATCH_API_USERS, headers=headers, timeout=5)
            if response.status_code in [429, 503]:
                write_error_log(f"{response.status_code} HTTP Error", "STAYWATCH_API_USERS", response.text[:100])
            response.raise_for_status()

            payload = response.json()
            if isinstance(payload, list):
                users = payload
            elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
                users = payload["data"]
            else:
                print("⚠️ 参加者APIレスポンスの形式が想定外のため、参加者候補なしで続行します")
                return ""

            lines = []
            for user in users:
                if isinstance(user, dict) and "id" in user and "name" in user:
                    lines.append(f"{user['id']}\t{user['name']}")

            print(f"👥 参加者候補をAPIから取得しました（{len(lines)}名）")
            return "\n".join(lines)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "503" in err_str:
                err_type = "429 Rate Limit" if "429" in err_str else "503 Service Unavailable"
                write_error_log(err_type, "STAYWATCH_API_USERS", err_str[:100])
            print(f"🔥 staywatch API (users) エラー（参加者候補なしで続行）: {e}")
            return ""

    def _create_prompt(self):
        """prompt.txt を読み込み、ゲーム候補と参加者候補を埋め込む"""
        if not os.path.exists(PROMPT_FILE):
            raise FileNotFoundError(f"{PROMPT_FILE} が見つかりません")

        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        game_candidates = self._get_game_candidates()
        if game_candidates is None:
            raise RuntimeError("digitalゲーム候補を取得できないため、Gemini推論を実行できません")

        member_candidates = self._get_member_candidates()
        if member_candidates is None:
            member_candidates = ""

        self.game_candidates_text = game_candidates
        self.member_candidates_text = member_candidates

        print(f"✅ {PROMPT_FILE} を読み込み、Promptを作成しました")
        return (
            prompt_template
            .replace("{GAME_CANDIDATES}", game_candidates)
            .replace("{MEMBER}", member_candidates)
        )

    def _open_camera(self):
        """カメラを開く"""
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ カメラ{i}に接続しました")
                return cap
            cap.release()
        raise RuntimeError("❌ カメラが見つかりません")

    # ======================================
    # 通信・外部操作
    # ======================================
    def _capture_image(self):
        """カメラから画像を1フレーム取得する（切断時は再接続）"""
        ret, frame = self.capture.read()
        if ret:
            return frame

        print("⚠️ カメラ再接続中...")
        self.capture.release()
        time.sleep(2)
        self.capture = self._open_camera()
        ret, frame = self.capture.read()

        if not ret:
            raise RuntimeError("❌ 画像取得失敗")
        return frame

    def _get_switch_state(self):
        """SwitchのON/OFF状態をAPIから取得 (packet, status_code)"""
        try:
            response = requests.get(self.SWITCH_API_URL, timeout=5)
            if response.status_code in [429, 503]:
                write_error_log(f"{response.status_code} HTTP Error", "SWITCH_API_URL", response.text[:100])
            response.raise_for_status()
            data = response.json()
            if "packet" not in data:
                raise ValueError("Switch APIレスポンスにpacketがありません")
            return bool(data["packet"]), 200
        except requests.RequestException as e:
            code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 500
            err_str = str(e)
            if code in [429, 503] or "429" in err_str or "503" in err_str:
                err_type = "429 Rate Limit" if (code == 429 or "429" in err_str) else "503 Service Unavailable"
                write_error_log(err_type, "SWITCH_API_URL", err_str[:100])
            print(f"❌ Switch状態取得エラー: {e}")
            return None, code
        except Exception as e:
            print(f"❌ Switch状態取得エラー: {e}")
            return None, 500

    def _send_result(self, class_id, member_ids=None, frame=None):
        """推定結果（ゲームID・複数参加者ID・ゲーム候補・参加者候補・画像）をAPIに送信"""
        try:
            member_ids_list = member_ids if member_ids else []
            member_id_str = ",".join(str(mid) for mid in member_ids_list)

            print(f"\n📤 結果送信 (class_id: {class_id}, member_id: {member_id_str or 'None'})")

            data = {
                "class_id": str(class_id),
                "member_id": member_id_str,
                "game_candidates": self.game_candidates_text or "",
                "member_candidates": self.member_candidates_text or "",
            }
            files = None

            if frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{class_id}_{timestamp}.jpg"
                _, img_encoded = cv2.imencode(".jpg", frame)
                files = {
                    "image": (filename, img_encoded.tobytes(), "image/jpeg")
                }

            response = requests.post(self.RESULT_API_URL, data=data, files=files, timeout=15)
            if response.status_code in [429, 503]:
                write_error_log(f"{response.status_code} HTTP Error", "RESULT_API_URL", response.text[:100])

            print(f"HTTP Status: {response.status_code}, Response: {response.text}")
            response.raise_for_status()
            return True, response.status_code
        except requests.RequestException as e:
            print(f"❌ 結果送信エラー: {e}")
            code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 500
            err_str = str(e)
            if code in [429, 503] or "429" in err_str or "503" in err_str:
                err_type = "429 Rate Limit" if (code == 429 or "429" in err_str) else "503 Service Unavailable"
                write_error_log(err_type, "RESULT_API_URL", err_str[:100])
            return False, code

    def _log_prediction(self, result, class_id, member_ids, confidence, reason):
        """結果をログファイルに保存"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        member_str = ",".join(str(m) for m in member_ids) if member_ids else "None"
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write("========================================\n")
                f.write(
                    f"日時: {timestamp}\n"
                    f"id: {class_id}\n"
                    f"member_id: {member_str}\n"
                    f"信頼度: {confidence}\n"
                    f"根拠: {reason}\n"
                )
                f.write(f"Gemini生回答:\n{result.strip() if result else 'Geminiから回答なし'}\n")
                f.write("========================================\n\n")
            print(f"📝 ログ保存: {LOG_FILE}")
        except Exception as e:
            print(f"⚠️ ログ保存エラー: {e}")

    # ======================================
    # Gemini 推論処理
    # ======================================
    def _recognize_boardgame_task(self, frame):
        """別スレッドで実行されるGemini推論の実体"""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        final_error_code = 500

        for model_name in FALLBACK_MODELS:
            try:
                print(f"🤖 Gemini 推論中 (モデル: {model_name})...")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=[image, self.prompt]
                )
                print(f"✅ Gemini API応答受信 (モデル: {model_name})")

                with self.gemini_lock:
                    if not response.text:
                        print(f"⚠️ {model_name} から回答がありません")
                        self.gemini_error_code = 500
                        write_error_log("Empty Response", model_name, "Gemini text response was empty")
                    else:
                        self.gemini_result = response.text
                        self.gemini_frame = frame
                    self.gemini_running = False
                return

            except Exception as e:
                error_str = str(e)
                code = 500

                if "429" in error_str:
                    code = 429
                    write_error_log("429 (Rate Limit Exceeded)", model_name, error_str[:100])
                elif "503" in error_str:
                    code = 503
                    write_error_log("503 (Service Unavailable)", model_name, error_str[:100])
                else:
                    if hasattr(e, "code"):
                        code = e.code
                    elif hasattr(e, "status_code"):
                        code = e.status_code
                    write_error_log(f"Gemini Error ({code})", model_name, error_str[:100])

                print(f"❌ Geminiエラー ({model_name}): {error_str} (Code: {code})")
                final_error_code = code

                if code == 503:
                    print("🔄 503エラーを検知しました。別のモデルで再試行します...")
                    time.sleep(2)
                    continue
                else:
                    break

        print("❌ すべてのモデルで推論に失敗したか、回復不能なエラーが発生しました。")
        with self.gemini_lock:
            self.gemini_error_code = final_error_code
            self.gemini_running = False

    def start_gemini_inference(self, frame):
        """推論スレッドの開始"""
        with self.gemini_lock:
            if self.gemini_running:
                return False
            self.gemini_running = True
            self.gemini_result = None
            self.gemini_frame = None
            self.gemini_error_code = None

        print("\n🔍 ゲーム・参加者推定開始...")
        threading.Thread(target=self._recognize_boardgame_task, args=(frame,), daemon=True).start()
        return True

    def get_gemini_result_if_done(self):
        """推論が完了していれば結果と画像フレームを返す (result, error_code, frame)"""
        with self.gemini_lock:
            if self.gemini_running:
                return None, None, None
            if self.gemini_error_code is not None:
                err_code = self.gemini_error_code
                self.gemini_error_code = None
                return None, err_code, None
            if self.gemini_result is not None:
                res = self.gemini_result
                frame = self.gemini_frame
                self.gemini_result = None
                self.gemini_frame = None
                return res, None, frame
        return None, None, None

    def _parse_member_ids(self, raw_val):
        """
        member_id文字列（カンマ区切り・単体・null等）またはリストを重複なしの list[int] に変換
        """
        if raw_val is None:
            return []

        if isinstance(raw_val, list):
            items = raw_val
        else:
            s = str(raw_val).strip().strip('"').strip("'").strip("[]")
            if s.lower() in ("", "null", "none", "nan", "不明", "なし", "〇〇"):
                return []
            # 全角カンマ・読点も半角カンマに統一して分割
            s = s.replace("、", ",").replace("，", ",")
            items = s.split(",")

        parsed_ids = []
        for item in items:
            token = str(item).strip().strip('"').strip("'")
            if not token or token.lower() in ("null", "none", "nan", "不明", "なし", "〇〇"):
                continue
            try:
                val = int(token)
                if val not in parsed_ids:
                    parsed_ids.append(val)
            except ValueError:
                continue

        return parsed_ids

    def parse_gemini_result(self, result):
        """JSONまたはテキストから (class_id, member_ids, confidence, reason) を抽出"""
        if not result:
            return None, [], None, None

        text = result.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1])

        try:
            data = json.loads(text)
            class_id = int(data.get("id")) if data.get("id") is not None else None
            member_ids = self._parse_member_ids(data.get("member_id"))
            confidence = data.get("confidence") or data.get("信頼度")
            reason = data.get("reason") or data.get("根拠")
            return class_id, member_ids, confidence, reason
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        class_id, member_ids, confidence, reason = None, [], None, None
        for line in result.splitlines():
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            sep = ":" if ":" in line_stripped else ("：" if "：" in line_stripped else None)
            if not sep:
                continue
            val = line_stripped.split(sep, 1)[1].strip()

            if line_lower.startswith("id:") or line_lower.startswith("id："):
                try:
                    class_id = int(val)
                except ValueError:
                    pass
            elif line_lower.startswith("member_id:") or line_lower.startswith("member_id："):
                member_ids = self._parse_member_ids(val)
            elif line_lower.startswith("信頼度:") or line_lower.startswith("信頼度："):
                confidence = val
            elif line_lower.startswith("根拠:") or line_lower.startswith("根拠："):
                reason = val

        return class_id, member_ids, confidence, reason

    # ======================================
    # 状態管理・メインロジック
    # ======================================
    def reset_prediction_state(self):
        self.inference_active = False
        self.inference_confirmed = False
        self.last_prediction_id = None
        self.last_member_ids = []
        self.same_prediction_count = 0
        self.next_action_time = 0
        print("🔄 推論状態をリセットしました")

    def schedule_next(self, interval):
        self.next_action_time = time.time() + interval

    def process_prediction(self, result, frame=None):
        """処理結果のステータスを文字列で返す"""
        class_id, member_ids, confidence, reason = self.parse_gemini_result(result)

        if class_id is None:
            print("⚠️ class_idを取得できませんでした")
            return "ERROR_500"

        member_display = ",".join(str(m) for m in member_ids) if member_ids else "None"
        print("================================")
        print(
            f"🎮 推定結果 | id: {class_id} | member_id: {member_display} "
            f"| 信頼度: {confidence} | 根拠: {reason}"
        )
        print("================================")
        self._log_prediction(result, class_id, member_ids, confidence, reason)

        # ID=0 (ゲームをしていない/メニュー画面など)
        if class_id == 0:
            success, code = self._send_result(0, member_ids=[], frame=frame)
            if success:
                self.last_prediction_id = 0
                self.last_member_ids = []
                self.same_prediction_count = 0
                self.inference_active = True  # SwitchがONの間はTrueを保持
                self.inference_confirmed = False
                self.schedule_next(NORMAL_INTERVAL)
                return "SUCCESS"
            return "ERROR_429" if code == 429 else "ERROR_500"

        # 初回推定 または ID変更
        if self.last_prediction_id != class_id:
            if self.last_prediction_id is None or self.last_prediction_id == 0:
                print(f"🎮 初回推定/再開始: id={class_id}, member_id={member_display}")
            else:
                print(f"🔄 ID変更: {self.last_prediction_id} → {class_id} (member_id: {member_display})")

            success, code = self._send_result(class_id, member_ids=member_ids, frame=frame)
            if success:
                self.last_prediction_id = class_id
                self.last_member_ids = list(member_ids)
                self.same_prediction_count = 1
                self.inference_active = True  # SwitchがONの間はTrueを保持
                self.schedule_next(NORMAL_INTERVAL)
                return "SUCCESS"
            return "ERROR_429" if code == 429 else "ERROR_500"

        # 同じゲームIDが連続した場合（member_idsが新たに確定・更新された場合も送信）
        self.same_prediction_count += 1
        print(f"🔁 同じID: {self.same_prediction_count}回連続")

        if member_ids and set(member_ids) != set(self.last_member_ids):
            prev_display = ",".join(str(m) for m in self.last_member_ids) if self.last_member_ids else "None"
            print(f"👤 参加者ID更新: {prev_display} → {member_display}")
            success, code = self._send_result(class_id, member_ids=member_ids, frame=frame)
            if success:
                self.last_member_ids = list(member_ids)
            elif code == 429:
                return "ERROR_429"

        if self.same_prediction_count >= 2:
            print("✅ ゲームを確定しました！ ⏰ 以降は長時間待機モードになります")
            self.inference_confirmed = True
            self.schedule_next(CONFIRMED_INTERVAL)
        else:
            self.schedule_next(NORMAL_INTERVAL)
        return "SUCCESS"

    def run(self):
        """メインループ"""
        try:
            while True:
                now = time.time()

                # 1. プレビュー表示 (常にスムーズに更新)
                frame = self._capture_image()
                cv2.imshow("Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # 2. Gemini推論中ならスキップ
                if self.gemini_running:
                    time.sleep(0.01)
                    continue

                # 3. Gemini推論結果の処理
                result, error_code, infer_frame = self.get_gemini_result_if_done()
                if result:
                    status = self.process_prediction(result, infer_frame)
                    if status == "ERROR_429":
                        print("⚠️ 429エラー(利用制限)を検知。1時間ストップします。")
                        self.schedule_next(RATE_LIMIT_INTERVAL)
                    elif status == "ERROR_500":
                        print("⚠️ 500エラー(サーバーエラー)を検知。10分ストップします。")
                        self.schedule_next(ERROR_INTERVAL)
                    continue
                elif error_code is not None:
                    if error_code == 429:
                        print("⚠️ Gemini 429エラー(高負荷/利用制限)。1時間ストップします。")
                        self.schedule_next(RATE_LIMIT_INTERVAL)
                    else:
                        print(f"⚠️ Gemini エラー (Code: {error_code})。10分ストップします。")
                        self.schedule_next(ERROR_INTERVAL)
                    continue

                # 4. Switch電源状態のチェック (指定間隔ごとに実行)
                if now - self.last_switch_check_time >= SWITCH_CHECK_INTERVAL:
                    self.last_switch_check_time = now
                    packet, sw_code = self._get_switch_state()

                    if packet is None:
                        if sw_code == 429:
                            print("⚠️ Switch API 429エラー。1時間ストップします。")
                            self.schedule_next(RATE_LIMIT_INTERVAL)
                        else:
                            print("⚠️ Switch API エラー。10分ストップします。")
                            self.schedule_next(ERROR_INTERVAL)
                    else:
                        # Switch: ON -> OFF
                        if not packet and self.last_packet:
                            print("🔌 Switch電源OFFを検知 ⚪ id=0を送信します")
                            success, code = self._send_result(0, member_ids=[])
                            if not success and code == 429:
                                self.schedule_next(RATE_LIMIT_INTERVAL)
                            self.reset_prediction_state()
                            self.last_packet = False

                        # Switch: OFF -> ON
                        elif not self.last_packet and packet:
                            print("🔄 Switch電源 ON を検知")
                            self.reset_prediction_state()
                            self.inference_active = True
                            self.last_packet = packet
                        else:
                            self.last_packet = packet

                # 5. 推論スケジュールの管理
                if self.inference_confirmed:
                    # 確定済みの長時間待機
                    if now >= self.next_action_time:
                        print("⏰ 長時間待機終了 🔍 再度ゲーム推論開始")
                        self.inference_confirmed = False
                        self.inference_active = True
                        self.last_prediction_id = None
                        self.last_member_ids = []
                        self.same_prediction_count = 0

                elif self.inference_active:
                    # 通常推論ターン (次回の予定時刻を超えていれば推論開始)
                    if now >= self.next_action_time:
                        self.start_gemini_inference(frame.copy())

                # メインループの短時間スリープ (GUI応答性維持のため)
                time.sleep(0.03)

        except KeyboardInterrupt:
            print("\n終了します")
        finally:
            if self.capture is not None:
                self.capture.release()
            cv2.destroyAllWindows()
            print("カメラを解放しました")


if __name__ == "__main__":
    app = GameRecognizerApp()
    app.run()