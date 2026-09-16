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

# インターバル設定（秒）
NORMAL_INTERVAL = 180         # 通常時：3分
CONFIRMED_INTERVAL = 3600     # ゲーム確定後：60分
ERROR_INTERVAL = 600          # 500エラー等：10分ストップ
RATE_LIMIT_INTERVAL = 3600    # 429エラー(Too Many Requests)時：1時間ストップ
SWITCH_CHECK_INTERVAL = 60    # Switch状態確認間隔：60秒

# フォールバック用のモデルリスト（上から順に試行します）
FALLBACK_MODELS = [
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
]

class GameRecognizerApp:
    def __init__(self):
        # 1. 環境変数の読み込みと検証
        load_dotenv()
        self.SWITCH_API_URL = self._get_env("SWITCH_API_URL")
        self.RESULT_API_URL = self._get_env("RESULT_API_URL")
        self.EVENTS_API_URL = self._get_env("EVENTS_API_URL")
        self.GEMINI_API_KEY = self._get_env("GEMINI_API_KEY")

        # 2. 初期化
        self.client = genai.Client(api_key=self.GEMINI_API_KEY)
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
        self.same_prediction_count = 0
        self.next_action_time = 0

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
    # 初期化関連
    # ======================================
    def _get_game_candidates(self):
        """APIからdigitalゲーム一覧を取得して文字列として返す"""
        try:
            response = requests.get(self.EVENTS_API_URL, params={"game_type": "digital"}, timeout=10)
            response.raise_for_status()
            games = response.json()["data"]

            print("🎮 digitalゲーム候補をAPIから取得しました")
            return "\n".join(f'    "{game["ID"]}" : "{game["Name"]}"' for game in games)
        except Exception as e:
            print(f"❌ digitalゲーム候補の取得に失敗しました: {e}")
            return None

    def _create_prompt(self):
        """Promptを作成する"""
        if not os.path.exists(PROMPT_FILE):
            raise FileNotFoundError(f"{PROMPT_FILE} が見つかりません")

        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        candidates = self._get_game_candidates()
        if candidates is None:
            raise RuntimeError("digitalゲーム候補を取得できないため、Gemini推論を実行できません")

        print("✅ prompt.txt を読み込み、Promptを作成しました")
        return prompt_template.replace("{GAME_CANDIDATES}", candidates)

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
            response.raise_for_status()
            data = response.json()
            if "packet" not in data:
                raise ValueError("Switch APIレスポンスにpacketがありません")
            return bool(data["packet"]), 200
        except requests.RequestException as e:
            code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 500
            print(f"❌ Switch状態取得エラー: {e}")
            return None, code
        except Exception as e:
            print(f"❌ Switch状態取得エラー: {e}")
            return None, 500

    def _send_result(self, class_id, frame=None):
        """推定結果と画像をAPIに送信し (Success(bool), status_code) を返す"""
        try:
            print(f"\n📤 結果送信 (class_id: {class_id})")
            
            data = {"class_id": str(class_id)}
            files = None

            # 画像フレームがある場合はエンコードして送信データに追加
            if frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{class_id}_{timestamp}.jpg"
                _, img_encoded = cv2.imencode(".jpg", frame)
                files = {
                    "image": (filename, img_encoded.tobytes(), "image/jpeg")
                }

            response = requests.post(self.RESULT_API_URL, data=data, files=files, timeout=15)
            print(f"HTTP Status: {response.status_code}, Response: {response.text}")
            response.raise_for_status()
            return True, response.status_code
        except requests.RequestException as e:
            print(f"❌ 結果送信エラー: {e}")
            code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 500
            return False, code

    def _log_prediction(self, result, class_id, confidence, reason):
        """結果をログファイルに保存"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write("========================================\n")
                f.write(f"日時: {timestamp}\nid: {class_id}\n信頼度: {confidence}\n根拠: {reason}\n")
                f.write(f"Gemini生回答:\n{result.strip() if result else 'Geminiから回答なし'}\n")
                f.write("========================================\n\n")
            print(f"📝 ログ保存: {LOG_FILE}")
        except Exception as e:
            print(f"⚠️ ログ保存エラー: {e}")

    # ======================================
    # Gemini 推論処理 (フォールバック対応)
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
                    else:
                        self.gemini_result = response.text
                        self.gemini_frame = frame
                    self.gemini_running = False
                
                # 成功した場合はメソッドを終了（次のモデルは試さない）
                return

            except Exception as e:
                error_str = str(e)
                code = 500
                
                # エラーコードの特定
                if "429" in error_str:
                    code = 429
                elif "503" in error_str:
                    code = 503
                elif hasattr(e, "code"):
                    code = e.code
                elif hasattr(e, "status_code"):
                    code = e.status_code

                print(f"❌ Geminiエラー ({model_name}): {error_str} (Code: {code})")
                final_error_code = code

                # 503エラーなら次のモデルで再試行
                if code == 503:
                    print(f"🔄 503エラーを検知しました。別のモデルで再試行します...")
                    time.sleep(2)  # サーバーへの連続アクセスを避けるための微小ウェイト
                    continue
                else:
                    # 503以外（429など）はフォールバックせずにループを抜ける
                    break

        # 全てのモデルで失敗したか、フォールバック対象外のエラーで抜けた場合
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

        print("\n🔍 ゲーム推定開始...")
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

    def parse_gemini_result(self, result):
        """JSONまたはテキストから結果を抽出"""
        if not result:
            return None, None, None

        text = result.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1])

        try:
            data = json.loads(text)
            return int(data.get("id")), data.get("confidence"), data.get("reason")
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        class_id, confidence, reason = None, None, None
        for line in result.splitlines():
            line_lower = line.strip().lower()
            if line_lower.startswith("id:") or line_lower.startswith("id："):
                try:
                    val = line.split(":", 1)[1] if ":" in line else line.split("：", 1)[1]
                    class_id = int(val.strip())
                except (ValueError, IndexError):
                    pass
            elif line_lower.startswith("信頼度:") or line_lower.startswith("信頼度："):
                try:
                    confidence = line.split(":", 1)[1] if ":" in line else line.split("：", 1)[1]
                except IndexError:
                    pass
            elif line_lower.startswith("根拠:") or line_lower.startswith("根拠："):
                try:
                    reason = line.split(":", 1)[1] if ":" in line else line.split("：", 1)[1]
                except IndexError:
                    pass

        return class_id, confidence.strip() if confidence else None, reason.strip() if reason else None

    # ======================================
    # 状態管理・メインロジック
    # ======================================
    def reset_prediction_state(self):
        self.inference_active = False
        self.inference_confirmed = False
        self.last_prediction_id = None
        self.same_prediction_count = 0
        self.next_action_time = 0
        print("🔄 推論状態をリセットしました")

    def schedule_next(self, interval):
        self.next_action_time = time.time() + interval

    def process_prediction(self, result, frame=None):
        """
        処理結果のステータスを文字列で返す
        "SUCCESS", "ERROR_429", "ERROR_500"
        """
        class_id, confidence, reason = self.parse_gemini_result(result)

        if class_id is None:
            print("⚠️ class_idを取得できませんでした")
            return "ERROR_500"

        print("================================")
        print(f"🎮 推定結果 | id: {class_id} | 信頼度: {confidence} | 根拠: {reason}")
        print("================================")
        self._log_prediction(result, class_id, confidence, reason)

        # ID=0 (ゲームをしていない/メニュー画面など)
        if class_id == 0:
            success, code = self._send_result(0, frame)
            if success:
                self.last_prediction_id = None
                self.same_prediction_count = 0
                self.inference_active = False
                self.inference_confirmed = False
                self.schedule_next(NORMAL_INTERVAL)
                return "SUCCESS"
            return "ERROR_429" if code == 429 else "ERROR_500"

        # 初回推定 または ID変更
        if self.last_prediction_id != class_id:
            if self.last_prediction_id is None:
                print(f"🎮 初回推定: id={class_id}")
            else:
                print(f"🔄 ID変更: {self.last_prediction_id} → {class_id}")
            
            success, code = self._send_result(class_id, frame)
            if success:
                self.last_prediction_id = class_id
                self.same_prediction_count = 1
                self.inference_active = False
                self.schedule_next(NORMAL_INTERVAL)
                return "SUCCESS"
            return "ERROR_429" if code == 429 else "ERROR_500"

        # 同じIDが連続した場合
        self.same_prediction_count += 1
        print(f"🔁 同じID: {self.same_prediction_count}回連続")

        if self.same_prediction_count >= 2:
            print("✅ ゲームを確定しました！ ⏰ 以降は長時間待機モードになります")
            self.inference_confirmed = True
            self.inference_active = False
            self.schedule_next(CONFIRMED_INTERVAL)
        else:
            self.inference_active = False
            self.schedule_next(NORMAL_INTERVAL)
        return "SUCCESS"

    def run(self):
        """メインループ"""
        try:
            while True:
                # 1. プレビュー表示
                frame = self._capture_image()
                cv2.imshow("Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # 2. Gemini推論中ならスキップして描画だけ回す
                if self.gemini_running:
                    time.sleep(0.01)
                    continue

                # 3. Gemini推論結果の処理
                result, error_code, infer_frame = self.get_gemini_result_if_done()
                
                # Gemini推論の成功時 (API送信の成否も含む)
                if result:
                    status = self.process_prediction(result, infer_frame)
                    if status == "ERROR_429":
                        print("⚠️ 429エラー(利用制限)を検知。1時間ストップします。")
                        self.schedule_next(RATE_LIMIT_INTERVAL)
                    elif status == "ERROR_500":
                        print("⚠️ 500エラー(サーバーエラー)を検知。10分ストップします。")
                        self.schedule_next(ERROR_INTERVAL)
                    continue
                
                # Gemini推論自体がエラーの場合
                elif error_code is not None:
                    if error_code == 429:
                        print("⚠️ Gemini 429エラー(高負荷/利用制限)。1時間ストップします。")
                        self.schedule_next(RATE_LIMIT_INTERVAL)
                    else:
                        print(f"⚠️ Gemini エラー (Code: {error_code})。10分ストップします。")
                        self.schedule_next(ERROR_INTERVAL)
                    continue

                # 4. Switch電源状態のチェック
                packet, sw_code = self._get_switch_state()
                if packet is None:
                    if sw_code == 429:
                        print("⚠️ Switch API 429エラー。1時間ストップします。")
                        self.schedule_next(RATE_LIMIT_INTERVAL)
                    else:
                        print("⚠️ Switch API エラー。10分ストップします。")
                        self.schedule_next(ERROR_INTERVAL)
                    time.sleep(SWITCH_CHECK_INTERVAL)
                    continue

                # Switch: ON -> OFF
                if not packet and self.last_packet:
                    print("🔌 Switch電源OFFを検知 ⚪ id=0を送信します")
                    success, code = self._send_result(0)
                    if not success and code == 429:
                        self.schedule_next(RATE_LIMIT_INTERVAL)
                    self.reset_prediction_state()
                    self.last_packet = False
                    time.sleep(SWITCH_CHECK_INTERVAL)
                    continue

                # Switch: OFF状態継続
                if not packet:
                    self.last_packet = False
                    time.sleep(SWITCH_CHECK_INTERVAL)
                    continue

                # Switch: OFF -> ON
                if not self.last_packet and packet:
                    print("🔄 Switch電源 ON を検知")
                    self.reset_prediction_state()
                    self.inference_active = True

                self.last_packet = packet

                # 5. 推論スケジュールの管理
                now = time.time()
                
                if self.inference_confirmed:
                    # 確定済みの長時間待機
                    if now >= self.next_action_time:
                        print("⏰ 長時間待機終了 🔍 再度ゲーム推論開始")
                        self.inference_confirmed = False
                        self.inference_active = True
                        self.last_prediction_id = None
                        self.same_prediction_count = 0
                
                elif self.inference_active:
                    # 通常推論ターン
                    if now >= self.next_action_time:
                        if not self.start_gemini_inference(frame.copy()):
                            print("⚠️ Gemini推論を開始できません")

                # ループウェイト（API叩きすぎ防止）
                time.sleep(SWITCH_CHECK_INTERVAL)

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