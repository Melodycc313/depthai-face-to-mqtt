# coding=utf-8
import os
import re
from time import monotonic
from pathlib import Path
from queue import Queue
import argparse
from time import sleep
import threading
import logging
import socket
import json
from datetime import datetime, timezone
from urllib import request as urllib_request
from urllib import error as urllib_error
from urllib.parse import quote

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import depthai
import numpy as np
from imutils.video import FPS
from face_control_server import run_server

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=["idle", "login", "server"],
    default="idle",
    help="idle: 不啟動相機；login: 單獨特徵測試；server: TCP 指令服務"
)

parser.add_argument("-p", "--preview", action="store_true",
                    help="preview camera")
parser.add_argument("-n", "--no-enroll", action="store_true",
                    help="compatibility flag; this version never enrolls")
parser.add_argument("-v", "--verbose", action="store_true")

args = parser.parse_args()

preview = args.preview
logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# ============================================================
# Face Preview -> Unity
# ============================================================
# Face 辨識期間由同一個 OAK-D pipeline 取得影像，
# 將目前正在辨識的 frame 壓成 JPEG 後送到 Unity。
# 不額外開第二個相機，也不碰 Pose 的 5004 / 5005 / 5006。
FACE_PREVIEW_HOST = "127.0.0.1"
FACE_PREVIEW_PORT = 5007
FACE_PREVIEW_FPS = 10.0
FACE_PREVIEW_JPEG_QUALITY = 65

# 讓 Unity 先顯示玩家自己的 FaceCameraPreview，
# 再開始接受登入比對結果，避免「先登入成功、後看到自己」。
FACE_LOGIN_PREVIEW_WARMUP_SECONDS = 2.0

# ============================================================
# Face Guide Gate
# ============================================================
# 只有臉位於中央安全區域、且大小合理時，才真正送進 ArcFace 比對。
# 座標皆以 0~1 表示相機畫面比例。
FACE_GUIDE_LEFT = 0.18
FACE_GUIDE_RIGHT = 0.82
FACE_GUIDE_TOP = 0.10
FACE_GUIDE_BOTTOM = 0.90

FACE_MIN_WIDTH_RATIO = 0.18
FACE_MIN_HEIGHT_RATIO = 0.22
FACE_MAX_WIDTH_RATIO = 0.72
FACE_MAX_HEIGHT_RATIO = 0.82

# ============================================================
# Face 模式舉手取消（MediaPipe PoseLandmarker）
# ============================================================
POSE_LANDMARKER_MODEL = (
    Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task"
)
FACE_CANCEL_POSE_FPS = 10.0
FACE_CANCEL_HOLD_SECONDS = 1.2
FACE_CANCEL_WRIST_ABOVE_NOSE_MARGIN = 0.035
FACE_CANCEL_MIN_VISIBILITY = 0.45

def to_planar(arr: np.ndarray, shape: tuple):
    return cv2.resize(arr, shape).transpose((2, 0, 1)).flatten()

def to_nn_result(nn_data):
    return np.array(nn_data.getFirstLayerFp16())

def run_nn(x_in, x_out, in_dict):
    nn_data = depthai.NNData()
    for key in in_dict:
        nn_data.setLayer(key, in_dict[key])
    x_in.send(nn_data)
    return x_out.tryGet()

def frame_norm(frame, *xy_vals):
    return (
        np.clip(np.array(xy_vals), 0, 1) * np.array(frame * (len(xy_vals) // 2))[::-1]
    ).astype(int)

def correction(frame, angle=None, invert=False):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1)
    affine = cv2.invertAffineTransform(mat).astype("float32")
    corr = cv2.warpAffine(
        frame,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if invert:
        return corr, affine
    return corr

# Experimental local biometric prototype; no face photographs are saved.
# This is NOT a production authentication system: threshold and anti-spoofing
# need validation before use with older adults or real accounts.
DATA_DIR = Path(__file__).resolve().parent / "face_profiles_gen2"
PROFILE_RE = re.compile(r"[A-Za-z0-9_-]{1,32}\Z")
SAMPLES_REQUIRED = 12
MATCHES_REQUIRED = 5
MATCH_THRESHOLD = 0.85   # Trial value, NOT a validated security threshold.
AMBIGUITY_MARGIN = 0.05
OPERATION_TIMEOUT = 35.0
SAMPLE_GAP_SECONDS = 0.35

# ============================================================
# Supabase biometric sync (server-side only)
# ============================================================
# The secret key MUST stay in Windows environment variables.
# Do not put it in Unity, source code, screenshots, or Git.
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "").strip()
SUPABASE_FACE_PHOTO_BUCKET = "face-photos"
SUPABASE_MODEL_NAME = "arcface"
SUPABASE_MODEL_VERSION = "face-recognition-mobilefacenet-arcface_2021.2_4shave"
SUPABASE_HTTP_TIMEOUT_SECONDS = 15.0
SUPABASE_PHOTO_JPEG_QUALITY = 90


def _supabase_is_configured():
    return bool(SUPABASE_URL and SUPABASE_SECRET_KEY)


def _supabase_request(method, path, payload=None, headers=None, raw_body=None):
    if not _supabase_is_configured():
        raise RuntimeError(
            "Supabase environment variables are missing: "
            "SUPABASE_URL / SUPABASE_SECRET_KEY"
        )

    url = SUPABASE_URL + path
    req_headers = {
        "apikey": SUPABASE_SECRET_KEY,
        "User-Agent": "GroceryStoreGame-FaceService/1.0",
    }

    if headers:
        req_headers.update(headers)

    body = raw_body
    if payload is not None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    req = urllib_request.Request(
        url=url,
        data=body,
        headers=req_headers,
        method=method,
    )

    try:
        with urllib_request.urlopen(
            req,
            timeout=SUPABASE_HTTP_TIMEOUT_SECONDS,
        ) as response:
            data = response.read()
            if not data:
                return None
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return json.loads(data.decode("utf-8"))
            return data
    except urllib_error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        raise RuntimeError(
            f"Supabase HTTP {exc.code} for {method} {path}: {detail}"
        ) from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(
            f"Supabase connection failed for {method} {path}: {exc}"
        ) from exc


def _supabase_create_face_profile(user_id):
    rows = _supabase_request(
        "POST",
        "/rest/v1/face_profiles?select=face_profile_id",
        payload={
            "user_id": int(user_id),
            "model_name": SUPABASE_MODEL_NAME,
            "model_version": SUPABASE_MODEL_VERSION,
            "is_active": False,
        },
        headers={
            "Prefer": "return=representation",
        },
    )

    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Supabase did not return face_profile_id")

    face_profile_id = rows[0].get("face_profile_id")
    if face_profile_id is None:
        raise RuntimeError("Supabase response missing face_profile_id")

    return int(face_profile_id)


def _supabase_insert_embeddings(face_profile_id, samples):
    rows = []
    for sample_index, vector in enumerate(samples):
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size != 128 or not np.all(np.isfinite(arr)):
            raise ValueError(
                f"invalid embedding at sample_index={sample_index}"
            )

        rows.append({
            "face_profile_id": int(face_profile_id),
            "sample_index": int(sample_index),
            "embedding": [float(v) for v in arr.tolist()],
        })

    if len(rows) != SAMPLES_REQUIRED:
        raise ValueError("cloud sync requires exactly 12 embeddings")

    _supabase_request(
        "POST",
        "/rest/v1/face_embeddings",
        payload=rows,
        headers={
            "Prefer": "return=minimal",
        },
    )


def _supabase_upload_profile_photo(user_id, face_profile_id, photo_bgr):
    if photo_bgr is None or photo_bgr.size == 0:
        raise ValueError("representative face photo is missing")

    ok, encoded = cv2.imencode(
        ".jpg",
        photo_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), SUPABASE_PHOTO_JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("failed to encode representative face photo")

    object_path = (
        f"user_{int(user_id)}/"
        f"profile_{int(face_profile_id)}.jpg"
    )
    encoded_path = quote(object_path, safe="/")

    _supabase_request(
        "POST",
        f"/storage/v1/object/{SUPABASE_FACE_PHOTO_BUCKET}/{encoded_path}",
        headers={
            "Content-Type": "image/jpeg",
            "x-upsert": "false",
        },
        raw_body=encoded.tobytes(),
    )

    return object_path


def _supabase_activate_profile(user_id, face_profile_id, photo_path):
    # First finish the new profile itself.
    _supabase_request(
        "PATCH",
        (
            "/rest/v1/face_profiles"
            f"?face_profile_id=eq.{int(face_profile_id)}"
        ),
        payload={
            "photo_path": photo_path,
            "is_active": True,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        headers={
            "Prefer": "return=minimal",
        },
    )

    # Only after the new profile is complete, deactivate older active profiles.
    _supabase_request(
        "PATCH",
        (
            "/rest/v1/face_profiles"
            f"?user_id=eq.{int(user_id)}"
            f"&is_active=eq.true"
            f"&face_profile_id=neq.{int(face_profile_id)}"
        ),
        payload={
            "is_active": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        headers={
            "Prefer": "return=minimal",
        },
    )


def _supabase_delete_profile(face_profile_id):
    try:
        _supabase_request(
            "DELETE",
            (
                "/rest/v1/face_profiles"
                f"?face_profile_id=eq.{int(face_profile_id)}"
            ),
            headers={
                "Prefer": "return=minimal",
            },
        )
    except Exception:
        logging.exception(
            "Failed to clean up incomplete Supabase face profile %s",
            face_profile_id,
        )


def sync_registration_to_supabase(user_id, samples, representative_photo):
    """
    Store one face profile, exactly 12 x 128-d embeddings, and one private
    representative face photo.

    This runs in a background thread so cloud latency cannot delay the
    OAK-D camera handoff or the existing local .npz login flow.
    """
    if not _supabase_is_configured():
        raise RuntimeError(
            "Supabase sync is not configured. "
            "Set SUPABASE_URL and SUPABASE_SECRET_KEY."
        )

    if not isinstance(user_id, str) or not user_id.isdigit():
        raise ValueError(
            "Supabase face sync requires a numeric user_id string"
        )

    cloud_samples = [
        np.asarray(v, dtype=np.float32).copy()
        for v in samples
    ]
    if len(cloud_samples) != SAMPLES_REQUIRED:
        raise ValueError("cloud sync requires exactly 12 samples")

    face_profile_id = None
    try:
        face_profile_id = _supabase_create_face_profile(user_id)
        _supabase_insert_embeddings(face_profile_id, cloud_samples)
        photo_path = _supabase_upload_profile_photo(
            user_id,
            face_profile_id,
            representative_photo,
        )
        _supabase_activate_profile(
            user_id,
            face_profile_id,
            photo_path,
        )
        logging.info(
            "Supabase face sync complete | user_id=%s | "
            "face_profile_id=%s | embeddings=%d | photo=%s",
            user_id,
            face_profile_id,
            len(cloud_samples),
            photo_path,
        )
        return face_profile_id
    except Exception:
        if face_profile_id is not None:
            _supabase_delete_profile(face_profile_id)
        raise


def load_profiles_from_supabase():
    """
    Load active biometric profiles from Supabase.

    Returns:
        dict[str, np.ndarray]: user_id -> shape (N, 128)

    Invalid/incomplete cloud profiles are skipped individually.
    A transport/API failure raises so the caller can fall back to local .npz.
    """
    if not _supabase_is_configured():
        raise RuntimeError(
            "Supabase sync is not configured. "
            "Set SUPABASE_URL and SUPABASE_SECRET_KEY."
        )

    profile_rows = _supabase_request(
        "GET",
        (
            "/rest/v1/face_profiles"
            "?select=face_profile_id,user_id"
            "&is_active=eq.true"
            "&order=face_profile_id.asc"
        ),
    )

    if not isinstance(profile_rows, list):
        raise RuntimeError("Invalid face_profiles response from Supabase")

    profiles = {}

    for row in profile_rows:
        try:
            face_profile_id = int(row["face_profile_id"])
            user_id = str(int(row["user_id"]))
        except (KeyError, TypeError, ValueError):
            logging.warning(
                "Skipping invalid Supabase face profile row: %r",
                row,
            )
            continue

        embedding_rows = _supabase_request(
            "GET",
            (
                "/rest/v1/face_embeddings"
                "?select=sample_index,embedding"
                f"&face_profile_id=eq.{face_profile_id}"
                "&order=sample_index.asc"
            ),
        )

        if not isinstance(embedding_rows, list):
            logging.warning(
                "Skipping cloud profile %s for user %s: "
                "invalid embedding response",
                face_profile_id,
                user_id,
            )
            continue

        vectors = []
        expected_index = 0
        valid_profile = True

        for item in embedding_rows:
            try:
                sample_index = int(item["sample_index"])
                raw_embedding = item["embedding"]
            except (KeyError, TypeError, ValueError):
                valid_profile = False
                break

            if sample_index != expected_index:
                valid_profile = False
                break

            vector = normalize_embedding(raw_embedding)
            if vector is None:
                valid_profile = False
                break

            vectors.append(vector)
            expected_index += 1

        # New registrations are expected to contain exactly 12 accepted samples.
        if not valid_profile or len(vectors) != SAMPLES_REQUIRED:
            logging.warning(
                "Skipping incomplete cloud profile %s for user %s: "
                "expected %d embeddings, got %d",
                face_profile_id,
                user_id,
                SAMPLES_REQUIRED,
                len(vectors),
            )
            continue

        profiles[user_id] = np.stack(vectors).astype(np.float32)

    return profiles


def load_profiles_cloud_first_with_local_fallback():
    """
    Gradual migration strategy:
    - Local .npz remains available for older users not yet uploaded.
    - Valid active Supabase profiles override the same user_id locally.
    - If Supabase is unavailable, login continues with local .npz only.
    """
    local_profiles = load_profiles()

    try:
        cloud_profiles = load_profiles_from_supabase()
    except Exception as exc:
        logging.warning(
            "Supabase face profile load failed; using local .npz fallback only: %s",
            exc,
        )
        if local_profiles:
            logging.info(
                "Face login profiles loaded from local fallback | users=%d",
                len(local_profiles),
            )
        return local_profiles

    if not cloud_profiles:
        if local_profiles:
            logging.warning(
                "No valid active Supabase face profiles; "
                "using local .npz fallback | users=%d",
                len(local_profiles),
            )
        return local_profiles

    merged = dict(local_profiles)
    merged.update(cloud_profiles)

    local_only_count = len(
        set(local_profiles.keys()) - set(cloud_profiles.keys())
    )

    logging.info(
        "Face login profiles ready | cloud=%d | local_only=%d | total=%d",
        len(cloud_profiles),
        local_only_count,
        len(merged),
    )

    return merged


def normalize_embedding(values):
    vec = np.asarray(values, dtype=np.float32).reshape(-1)
    if vec.size != 128 or not np.all(np.isfinite(vec)):
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return None
    return vec / norm


def profile_path(user_id):
    if not isinstance(user_id, str) or not PROFILE_RE.fullmatch(user_id):
        raise ValueError("user_id must be 1-32 ASCII letters, digits, _ or -")
    return DATA_DIR / (user_id + ".npz")


def load_profiles():
    if not DATA_DIR.exists():
        return {}
    profiles = {}
    for path in DATA_DIR.glob("*.npz"):
        try:
            if not PROFILE_RE.fullmatch(path.stem):
                continue
            with np.load(path, allow_pickle=False) as record:
                samples = np.asarray(record["embeddings"], dtype=np.float32)
            if samples.ndim != 2 or samples.shape[1] != 128 or len(samples) < 1:
                raise ValueError("invalid feature shape")
            valid = [normalize_embedding(v) for v in samples]
            if any(v is None for v in valid):
                raise ValueError("invalid values")
            profiles[path.stem] = np.stack(valid)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logging.warning("Skipping invalid profile %s: %s", path.name, exc)
    return profiles


def save_profile_once(user_id, samples):
    path = profile_path(user_id)
    arr = np.stack(samples).astype(np.float32)
    if arr.shape != (SAMPLES_REQUIRED, 128):
        raise ValueError("invalid enrollment sample count")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Exclusive creation ensures an existing account cannot be overwritten.
    with path.open("xb") as handle:
        try:
            np.savez_compressed(handle, embeddings=arr)
        except BaseException:
            handle.close()
            path.unlink(missing_ok=True)
            raise


class DepthAI:
    def __init__(
        self
    ):
        logging.debug("Loading pipeline...")
        self.fps_cam = FPS()
        self.fps_nn = FPS()
        self.create_pipeline()
        try:
            self.start_pipeline()
        except Exception:
            # If queue setup fails after opening USB, release the partial device.
            if hasattr(self, "device"):
                self.device.close()
            raise
        self.fontScale = 1
        self.lineType = 0

        # Face Preview UDP sender.
        # 只在 Face pipeline 實際持有 OAK-D 時存在。
        self._preview_udp_socket = None
        self._last_preview_send_at = 0.0

        # Face 模式下的舉手取消。
        # 使用同一張 OAK-D frame，不會再開第二個相機 pipeline。
        self.on_cancel_gesture = None
        self._pose_landmarker = None
        self._last_pose_process_at = 0.0
        self._cancel_raise_started_at = None
        self._cancel_wait_for_release = True
        self._cancel_confirmed = False

        self._create_pose_landmarker()

    def _create_pose_landmarker(self):
        if not POSE_LANDMARKER_MODEL.exists():
            logging.warning(
                "PoseLandmarker model not found: %s",
                POSE_LANDMARKER_MODEL,
            )
            return

        try:
            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(POSE_LANDMARKER_MODEL)
                ),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._pose_landmarker = (
                mp_vision.PoseLandmarker.create_from_options(options)
            )
            logging.info(
                "MediaPipe PoseLandmarker ready for Face cancel gesture."
            )
        except Exception:
            logging.exception(
                "Failed to initialize MediaPipe PoseLandmarker; "
                "Face recognition will continue without raise-hand cancel."
            )
            self._pose_landmarker = None

    def _close_pose_landmarker(self):
        if self._pose_landmarker is None:
            return
        try:
            self._pose_landmarker.close()
        except Exception:
            logging.exception("Failed to close PoseLandmarker")
        finally:
            self._pose_landmarker = None

    def _report_cancel_gesture(self, progress, confirmed=False):
        if self.on_cancel_gesture is not None:
            self.on_cancel_gesture(
                float(np.clip(progress, 0.0, 1.0)),
                bool(confirmed),
            )

    def _reset_cancel_gesture(self):
        self._cancel_raise_started_at = None
        if not self._cancel_confirmed:
            self._report_cancel_gesture(0.0, False)

    @staticmethod
    def _landmark_visible(landmark):
        visibility = getattr(landmark, "visibility", 1.0)
        presence = getattr(landmark, "presence", 1.0)
        return (
            visibility >= FACE_CANCEL_MIN_VISIBILITY
            and presence >= FACE_CANCEL_MIN_VISIBILITY
        )

    def _process_cancel_gesture(self):
        """
        Face 擁有 OAK-D 時，直接從 Face pipeline 的同一張 frame
        判斷玩家是否把任一手腕舉到鼻子上方。
        """
        if self._pose_landmarker is None or self._cancel_confirmed:
            return

        now = monotonic()
        min_interval = 1.0 / max(1.0, FACE_CANCEL_POSE_FPS)
        if now - self._last_pose_process_at < min_interval:
            return
        self._last_pose_process_at = now

        try:
            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            result = self._pose_landmarker.detect(mp_image)
        except Exception:
            logging.exception("PoseLandmarker detect failed")
            self._reset_cancel_gesture()
            return

        if not result.pose_landmarks:
            # 沒有人時進度歸零；同時視為已經「放下手」。
            self._cancel_wait_for_release = False
            self._reset_cancel_gesture()
            return

        landmarks = result.pose_landmarks[0]
        nose = landmarks[0]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        nose_ok = self._landmark_visible(nose)
        left_ok = self._landmark_visible(left_wrist)
        right_ok = self._landmark_visible(right_wrist)

        left_raised = (
            nose_ok
            and left_ok
            and left_wrist.y
            < nose.y - FACE_CANCEL_WRIST_ABOVE_NOSE_MARGIN
        )
        right_raised = (
            nose_ok
            and right_ok
            and right_wrist.y
            < nose.y - FACE_CANCEL_WRIST_ABOVE_NOSE_MARGIN
        )
        any_raised = left_raised or right_raised

        # 玩家通常剛在 FaceConsent 用舉手確認；
        # 必須先放下，再重新舉手，才允許取消。
        if self._cancel_wait_for_release:
            if not any_raised:
                self._cancel_wait_for_release = False
                self._reset_cancel_gesture()
            return

        if not any_raised:
            self._reset_cancel_gesture()
            return

        if self._cancel_raise_started_at is None:
            self._cancel_raise_started_at = now
            logging.info(
                "Raised hand detected in Face mode; "
                "starting %.1f s cancel hold.",
                FACE_CANCEL_HOLD_SECONDS,
            )

        progress = float(np.clip(
            (now - self._cancel_raise_started_at)
            / max(0.2, FACE_CANCEL_HOLD_SECONDS),
            0.0,
            1.0,
        ))

        if progress >= 1.0:
            self._cancel_confirmed = True
            self._report_cancel_gesture(1.0, True)
            logging.info("Face cancel gesture confirmed by raised hand.")
        else:
            self._report_cancel_gesture(progress, False)

    def create_pipeline(self):
        logging.debug("Creating pipeline...")
        self.pipeline = depthai.Pipeline()

        # ColorCamera
        logging.debug("Creating Color Camera...")
        self.cam = self.pipeline.createColorCamera()
        self.cam.setPreviewSize(self._cam_size[1], self._cam_size[0])
        self.cam.setResolution(
            depthai.ColorCameraProperties.SensorResolution.THE_4_K
        )
        self.cam.setInterleaved(False)
        self.cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
        self.cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.BGR)

        self.cam_xout = self.pipeline.createXLinkOut()
        self.cam_xout.setStreamName("preview")
        self.cam.preview.link(self.cam_xout.input)

        self.create_nns()

        logging.info("Pipeline created.")

    def create_nns(self):
        pass

    def create_nn(self, model_path: str, model_name: str, first: bool = False):
        """

        :param model_path: model path
        :param model_name: model abbreviation
        :param first: Is it the first model
        :return:
        """
        # NeuralNetwork
        logging.debug(f"Creating {model_path} Neural Network...")
        model_nn = self.pipeline.createNeuralNetwork()
        model_nn.setBlobPath(str(Path(f"{model_path}").resolve().absolute()))
        model_nn.input.setBlocking(False)
        if first:
            logging.debug("linked cam.preview to model_nn.input")
            self.cam.preview.link(model_nn.input)
        else:
            model_in = self.pipeline.createXLinkIn()
            model_in.setStreamName(f"{model_name}_in")
            model_in.out.link(model_nn.input)

        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    def create_mobilenet_nn(
        self,
        model_path: str,
        model_name: str,
        conf: float = 0.5,
        first: bool = False,
    ):
        """

        :param model_path: model name
        :param model_name: model abbreviation
        :param conf: confidence threshold
        :param first: Is it the first model
        :return:
        """
        # NeuralNetwork
        logging.debug(f"Creating {model_path} MobileNet Neural Network...")
        model_nn = self.pipeline.createMobileNetDetectionNetwork()
        model_nn.setBlobPath(str(Path(f"{model_path}").resolve().absolute()))
        model_nn.setConfidenceThreshold(conf)
        model_nn.input.setBlocking(False)

        if first:
            self.cam.preview.link(model_nn.input)
        else:
            model_in = self.pipeline.createXLinkIn()
            model_in.setStreamName(f"{model_name}_in")
            model_in.out.link(model_nn.input)

        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    def start_pipeline(self):
        logging.info("Starting pipeline...")
        self.device = depthai.Device(self.pipeline)
        self.start_nns()

        self.preview = self.device.getOutputQueue(
            name="preview", maxSize=4, blocking=False
        )

    def start_nns(self):
        pass

    def put_text(self, text, dot, color=(0, 0, 255), font_scale=None,
        line_type=None):
        font_scale = font_scale if font_scale else self.fontScale
        line_type = line_type if line_type else self.lineType
        dot = tuple(dot[:2])
        cv2.putText(
            img=self.debug_frame,
            text=text,
            org=dot,
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=font_scale,
            color=color,
            lineType=line_type,
        )

    def draw_bbox(self, bbox, color):
        cv2.rectangle(
            img=self.debug_frame,
            pt1=(bbox[0], bbox[1]),
            pt2=(bbox[2], bbox[3]),
            color=color,
            thickness=2,
        )

    def parse(self):
        if preview:
            self.debug_frame = self.frame.copy()

        s = self.parse_fun()
        # if s :
        #     raise StopIteration()
        if preview:
            cv2.imshow(
                "Camera_view",
                self.debug_frame,
            )
            self.fps_cam.update()
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                self.fps_cam.stop()
                self.fps_nn.stop()
                logging.debug(
                    f"FPS_CAMERA: {self.fps_cam.fps():.2f} , FPS_NN: {self.fps_nn.fps():.2f}"
                )
                raise StopIteration()

    def _ensure_preview_socket(self):
        """建立 Face Preview UDP socket；失敗時只停用預覽，不中斷辨識。"""
        if self._preview_udp_socket is not None:
            return True

        try:
            self._preview_udp_socket = socket.socket(
                socket.AF_INET,
                socket.SOCK_DGRAM,
            )
            return True
        except OSError as exc:
            logging.warning("Face Preview socket 建立失敗：%s", exc)
            self._preview_udp_socket = None
            return False

    def _send_preview_frame(self):
        """將目前 frame 以 JPEG UDP 傳給 Unity 的 FaceCameraPreview。"""
        if not hasattr(self, "frame") or self.frame is None:
            return

        now = monotonic()
        min_interval = 1.0 / max(1.0, FACE_PREVIEW_FPS)

        if now - self._last_preview_send_at < min_interval:
            return

        if not self._ensure_preview_socket():
            return

        try:
            ok, encoded = cv2.imencode(
                ".jpg",
                self.frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), FACE_PREVIEW_JPEG_QUALITY],
            )

            if not ok:
                return

            payload = encoded.tobytes()

            # 單一 UDP datagram 理論上限約 65 KB。
            # 目前 Face preview 為 300x300，quality=65 通常遠低於此上限。
            if len(payload) > 60000:
                logging.warning(
                    "Face Preview JPEG 過大（%d bytes），本張略過",
                    len(payload),
                )
                return

            self._preview_udp_socket.sendto(
                payload,
                (FACE_PREVIEW_HOST, FACE_PREVIEW_PORT),
            )
            self._last_preview_send_at = now

        except OSError as exc:
            # Preview 失敗不應讓人臉辨識本身失敗。
            logging.debug("Face Preview UDP 傳送失敗：%s", exc)

    def _close_preview_socket(self):
        if self._preview_udp_socket is None:
            return

        try:
            self._preview_udp_socket.close()
        except OSError:
            pass
        finally:
            self._preview_udp_socket = None

    def run_camera(self, stop_event=None):
        while stop_event is None or not stop_event.is_set():
            in_rgb = self.preview.tryGet()
            if in_rgb is None:
                sleep(0.01)
                continue
            if stop_event is None or not stop_event.is_set():
                shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
                self.frame = (
                    in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                )
                self.frame = np.ascontiguousarray(self.frame)

                # 直接使用 Face pipeline 正在辨識的同一張 frame。
                # 即使目前沒有偵測到臉，玩家仍能在 Unity 看見自己並調整位置。
                self._send_preview_frame()

                # Face 持有相機時，用同一張 frame 偵測「舉手取消」。
                self._process_cancel_gesture()

                try:
                    self.parse()
                except StopIteration:
                    break

    @property
    def cam_size(self):
        return self._cam_size

    @cam_size.setter
    def cam_size(self, v):
        self._cam_size = v


    def run(self, stop_event=None):
        self.fps_cam.start()
        self.fps_nn.start()

        try:
            self.run_camera(stop_event)
        finally:
            # 先關閉 CPU 端 PoseLandmarker / Preview，再釋放 OAK-D。
            self._close_pose_landmarker()
            self._close_preview_socket()

            logging.info("Closing OAK-D device...")
            try:
                self.device.close()
            finally:
                if preview:
                    cv2.destroyAllWindows()
            logging.info("OAK-D device closed.")


class Main(DepthAI):
    def __init__(self):
        self.cam_size = (300, 300)
        super(Main, self).__init__()
        self.face_frame_corr = Queue()
        self.face_frame = Queue()
        self.face_coords = Queue()
        self.embedding_count = 0
        self.on_embedding = None

        # 回報目前玩家是否已進入臉部引導框。
        # 狀態：waiting_face / multiple_faces / adjust_position / running
        self.on_face_state = None
        self.stop_event = threading.Event()

    def create_nns(self):

        self.create_mobilenet_nn(
            "models/face-detection-retail-0005_openvino_2021.4_4shave.blob",
            "mfd",
            first=True,
            conf=0.9, # Raised to prevent auto-enroll of non-faces
        )

        self.create_nn(
            "models/head-pose-estimation-adas-0001_openvino_2021.4_4shave.blob",
            "head_pose",
        )
        self.create_nn(
            "models/face-recognition-mobilefacenet-arcface_2021.2_4shave.blob",
            "arcface",
        )

    def start_nns(self):
        self.mfd_nn = self.device.getOutputQueue("mfd_nn", 4, False)
        self.head_pose_in = self.device.getInputQueue("head_pose_in", 4, False)
        self.head_pose_nn = self.device.getOutputQueue("head_pose_nn", 4, False)
        self.arcface_in = self.device.getInputQueue("arcface_in", 4, False)
        self.arcface_nn = self.device.getOutputQueue("arcface_nn", 4, False)

    def _report_face_state(self, face_state):
        if self.on_face_state is not None:
            self.on_face_state(face_state)

    def run_face_mn(self):
        nn_data = self.mfd_nn.tryGet()
        if nn_data is None:
            return False

        bboxes = nn_data.detections

        if len(bboxes) == 0:
            self._report_face_state("waiting_face")
            return False

        # 登入 / 註冊都只允許一張臉。
        if len(bboxes) != 1:
            self._report_face_state("multiple_faces")
            return False

        bbox = bboxes[0]

        if self.stop_event.is_set():
            return False

        # 先直接使用 detector 的 0~1 座標判斷是否位於中央引導區。
        xmin = float(np.clip(bbox.xmin, 0.0, 1.0))
        ymin = float(np.clip(bbox.ymin, 0.0, 1.0))
        xmax = float(np.clip(bbox.xmax, 0.0, 1.0))
        ymax = float(np.clip(bbox.ymax, 0.0, 1.0))

        face_w = xmax - xmin
        face_h = ymax - ymin

        inside_guide = (
            xmin >= FACE_GUIDE_LEFT and
            xmax <= FACE_GUIDE_RIGHT and
            ymin >= FACE_GUIDE_TOP and
            ymax <= FACE_GUIDE_BOTTOM
        )

        size_ok = (
            FACE_MIN_WIDTH_RATIO <= face_w <= FACE_MAX_WIDTH_RATIO and
            FACE_MIN_HEIGHT_RATIO <= face_h <= FACE_MAX_HEIGHT_RATIO
        )

        if not inside_guide or not size_ok:
            self._report_face_state("adjust_position")
            return False

        # 到這裡才代表玩家真的在框框內，可以開始辨識。
        self._report_face_state("running")

        face_coord = frame_norm(
            self.frame.shape[:2],
            *[bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
        )

        crop = self.frame[
            face_coord[1]:face_coord[3],
            face_coord[0]:face_coord[2]
        ]

        if crop.size == 0:
            self._report_face_state("adjust_position")
            return False

        self.face_frame.put(crop)
        self.face_coords.put(face_coord)

        if preview:
            self.draw_bbox(face_coord, (10, 245, 10))

        return True

    def run_head_pose(self):
        while self.face_frame.qsize() and not self.stop_event.is_set():
            face_frame = self.face_frame.get()
            nn_data = run_nn(
                self.head_pose_in,
                self.head_pose_nn,
                {"data": to_planar(face_frame, (60, 60))},
            )
            if nn_data is None:
                return False

            out = np.array(nn_data.getLayerFp16("angle_r_fc"))
            self.face_frame_corr.put(correction(face_frame, -out[0]))

        return True

    def run_arcface(self):
        while self.face_frame_corr.qsize() and not self.stop_event.is_set():
            face_coords = self.face_coords.get()
            face_frame = self.face_frame_corr.get()

            nn_data = run_nn(
                self.arcface_in,
                self.arcface_nn,
                {"data": to_planar(face_frame, (112, 112))},
            )

            if nn_data is None:
                return False
            self.fps_nn.update()
            results = to_nn_result(nn_data)

            self.embedding_count += 1

            if self.embedding_count == 1 or self.embedding_count % 30 == 0:
                logging.info(
                    "ArcFace OK | count=%d | dimensions=%d | finite=%s | norm=%.4f",
                    self.embedding_count,
                    results.size,
                    np.isfinite(results).all(),
                    np.linalg.norm(results),
                )

            if results.size == 128 and np.isfinite(results).all() and np.linalg.norm(results) > 0:
                if self.on_embedding is not None:
                    self.on_embedding(
                        self.embedding_count,
                        results,
                        face_frame.copy(),
                    )
            else:
                logging.warning("Invalid ArcFace output ignored")

            if preview:
                self.put_text(
                    f"ArcFace samples: {self.embedding_count}",
                    (face_coords[0], max(20, face_coords[1] - 10)),
                    (244, 0, 255),
                )

        return True

    def parse_fun(self):
        if self.run_face_mn():
            if self.run_head_pose():
                if self.run_arcface():
                    return True

class FaceController:
    """One camera owner; explicit consent; prototype enrollment and matching."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = None
        self.state = "idle"
        self.stop_target = "idle"
        self.embedding_count = 0
        self.last_error = None
        self.authenticated = False
        self.user_id = None
        self.operation = None
        self.enroll_user = None
        self.samples = []
        self.last_sample_at = 0.0
        self.best_register_photo = None
        self.best_register_photo_score = -1.0
        self.cloud_sync_state = "idle"
        self.cloud_sync_error = None
        self.cloud_face_profile_id = None
        self.cloud_sync_worker = None
        self.deadline = 0.0
        self.match_name = None
        self.match_streak = 0
        self.profiles = {}

        # Login UX：Face 相機開始後，先讓 Unity 預覽一小段時間，
        # 再允許人臉比對成功。
        self.login_match_allowed_at = 0.0

        # True only when the Face pipeline no longer owns OAK-D.
        # Unity should wait for this before sending POSE_START.
        self.camera_released = True

        # Face 模式 MediaPipe 舉手取消狀態，供 Unity status 輪詢。
        self.cancel_progress = 0.0
        self.cancel_gesture_confirmed = False

    def status(self):
        with self.lock:
            result = {"ok": True, "state": self.state,
                      "embedding_count": self.embedding_count,
                      "authenticated": self.authenticated,
                      "user_id": self.user_id, "error": self.last_error,
                      "camera_released": self.camera_released,
                      "cancel_progress": self.cancel_progress,
                      "cancel_gesture_confirmed": self.cancel_gesture_confirmed,
                      "cloud_sync_state": self.cloud_sync_state,
                      "cloud_sync_error": self.cloud_sync_error,
                      "cloud_face_profile_id": self.cloud_face_profile_id}
            if self.operation == "register" and self.state in ("starting", "running", "stopping"):
                result["samples_collected"] = len(self.samples)
                result["samples_required"] = SAMPLES_REQUIRED
            return result

    def handle(self, request):
        if not isinstance(request, dict):
            return {"ok": False, "error": "Request must be a JSON object"}
        command = request.get("command")
        if command == "ping":
            return {"ok": True, "message": "pong"}
        if command == "status":
            return self.status()
        if command in ("register", "login"):
            if request.get("consent") is not True:
                return {"ok": False, "error": "Explicit consent required"}
            user_id = request.get("user_id") if command == "register" else None
            if command == "register":
                try:
                    path = profile_path(user_id)
                except ValueError as exc:
                    return {"ok": False, "error": str(exc)}
                if path.exists():
                    return {"ok": False, "error": "user_id already exists; will not overwrite"}
            else:
                profiles = load_profiles_cloud_first_with_local_fallback()
                if not profiles:
                    return {
                        "ok": False,
                        "error": (
                            "No enrolled profiles available from Supabase "
                            "or local .npz; register with consent first"
                        ),
                    }
            with self.lock:
                if self.state in ("starting", "running", "stopping") or (
                    self.worker is not None and self.worker.is_alive()
                ):
                    return {"ok": False, "state": self.state, "error": "Camera busy"}
                if command == "register" and profile_path(user_id).exists():
                    return {"ok": False, "error": "user_id already exists; will not overwrite"}
                self.stop_event = threading.Event()
                self.stop_target = "idle"
                self.embedding_count = 0
                self.last_error = None
                self.authenticated = False
                self.user_id = None
                self.operation = command
                self.enroll_user = user_id if command == "register" else None
                self.samples = []
                self.last_sample_at = 0.0
                self.best_register_photo = None
                self.best_register_photo_score = -1.0
                if command == "register":
                    self.cloud_sync_state = "waiting"
                    self.cloud_sync_error = None
                    self.cloud_face_profile_id = None
                self.deadline = monotonic() + OPERATION_TIMEOUT
                self.match_name = None
                self.match_streak = 0
                self.profiles = profiles if command == "login" else {}
                self.cancel_progress = 0.0
                self.cancel_gesture_confirmed = False

                # 不能在收到 login 指令時就開始計時：
                # OAK-D pipeline 本身需要數秒啟動，若此時開始，
                # 等真正看到玩家時 warm-up 早就結束了。
                #
                # 改成等玩家「第一次真正進入臉部框」時，
                # 才由 _process_face_state() 開始倒數。
                self.login_match_allowed_at = 0.0

                # From this point onward Face owns, or is about to own, the OAK-D.
                # Do not let Pose restart until _camera_worker finally marks it released.
                self.camera_released = False
                self.state = "starting"
                self.worker = threading.Thread(target=self._camera_worker,
                                               args=(self.stop_event,),
                                               name="OakFaceWorker", daemon=False)
                self.worker.start()
            return {"ok": True, "state": "starting", "operation": command,
                    "message": "Experimental biometric prototype; not production authentication"}
        if command in ("cancel", "guest"):
            target = "guest" if command == "guest" else "idle"
            with self.lock:
                self.authenticated = False
                self.user_id = None
                self.samples = []
                self.cancel_progress = 0.0
                self.cancel_gesture_confirmed = False
                if self.state in ("starting", "running", "stopping",
                                  "waiting_face", "multiple_faces",
                                  "adjust_position") or (
                    self.worker is not None and self.worker.is_alive()
                ):
                    self.stop_target = target
                    self.stop_event.set()
                    self.state = "stopping"
                else:
                    self.state = target
                    self.last_error = None
                    self.operation = None
                    self.camera_released = True
                state = self.state
            return {"ok": True, "state": state,
                    "message": "Wait for status=guest/idle before starting another mode"}
        return {"ok": False, "error": "Unknown command"}

    def _process_cancel_gesture(self, progress, confirmed):
        with self.lock:
            if self.operation != "login":
                self.cancel_progress = 0.0
                self.cancel_gesture_confirmed = False
                return

            if self.state in (
                "matched_experimental",
                "registered",
                "timeout",
                "error",
                "stopping",
                "idle",
                "guest",
            ):
                return

            self.cancel_progress = float(
                np.clip(progress, 0.0, 1.0)
            )

            if confirmed:
                self.cancel_gesture_confirmed = True

    def _process_face_state(self, face_state):
        """
        Main 回報玩家是否已進入引導框。

        login 的辨識延遲必須從「真正進入框框」才開始，
        不能從 login command 被接受時開始，因為 OAK-D pipeline
        啟動本身可能就需要數秒。
        """
        allowed_states = {
            "waiting_face",
            "multiple_faces",
            "adjust_position",
            "running",
        }

        if face_state not in allowed_states:
            return

        with self.lock:
            if self.stop_event.is_set():
                return

            if self.operation not in ("login", "register"):
                return

            if self.state in (
                "matched_experimental",
                "registered",
                "timeout",
                "error",
                "stopping",
            ):
                return

            previous_state = self.state

            if self.operation == "login":
                if face_state == "running":
                    # 只有從「不在正確位置」切換到「位置正確」時，
                    # 才開始一次新的辨識延遲。
                    if previous_state != "running":
                        self.login_match_allowed_at = (
                            monotonic()
                            + FACE_LOGIN_PREVIEW_WARMUP_SECONDS
                        )
                        self.match_name = None
                        self.match_streak = 0
                        logging.info(
                            "Face is inside guide; recognition will start "
                            "after %.1f s. Raise hand during this window to cancel.",
                            FACE_LOGIN_PREVIEW_WARMUP_SECONDS,
                        )
                else:
                    # 一離開框框，就清除比對累積。
                    # 下次重新進框會重新取得完整的取消時間。
                    self.login_match_allowed_at = 0.0
                    self.match_name = None
                    self.match_streak = 0

            self.state = face_state

    @staticmethod
    def _photo_quality_score(photo_bgr):
        if photo_bgr is None or photo_bgr.size == 0:
            return -1.0
        try:
            gray = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            h, w = gray.shape[:2]
            area_bonus = float(h * w) / 1000.0
            return sharpness + area_bonus
        except Exception:
            return -1.0

    def _start_supabase_sync(self, user_id, samples, representative_photo):
        if not _supabase_is_configured():
            with self.lock:
                self.cloud_sync_state = "error"
                self.cloud_sync_error = (
                    "SUPABASE_URL / SUPABASE_SECRET_KEY missing"
                )
            logging.error(
                "Local face profile saved, but Supabase sync is not configured."
            )
            return

        cloud_samples = [
            np.asarray(v, dtype=np.float32).copy()
            for v in samples
        ]
        cloud_photo = (
            representative_photo.copy()
            if representative_photo is not None
            else None
        )

        def worker():
            with self.lock:
                self.cloud_sync_state = "syncing"
                self.cloud_sync_error = None
            try:
                profile_id = sync_registration_to_supabase(
                    user_id,
                    cloud_samples,
                    cloud_photo,
                )
                with self.lock:
                    self.cloud_face_profile_id = profile_id
                    self.cloud_sync_state = "complete"
                    self.cloud_sync_error = None
            except Exception as exc:
                logging.exception(
                    "Supabase face sync failed for user_id=%s",
                    user_id,
                )
                with self.lock:
                    self.cloud_sync_state = "error"
                    self.cloud_sync_error = str(exc)

        thread = threading.Thread(
            target=worker,
            name=f"SupabaseFaceSync-{user_id}",
            daemon=True,
        )
        self.cloud_sync_worker = thread
        thread.start()

    def _process_embedding(self, count, raw, face_photo=None):
        vector = normalize_embedding(raw)
        if vector is None:
            return
        with self.lock:
            if self.stop_event.is_set() or self.state != "running":
                return
            self.embedding_count = count
            if monotonic() > self.deadline:
                self.state = "timeout"
                self.stop_event.set()
                return
            if self.operation == "register":
                if monotonic() - self.last_sample_at < SAMPLE_GAP_SECONDS:
                    return
                self.last_sample_at = monotonic()
                self.samples.append(vector.copy())

                photo_score = self._photo_quality_score(face_photo)
                if photo_score > self.best_register_photo_score:
                    self.best_register_photo_score = photo_score
                    self.best_register_photo = (
                        face_photo.copy()
                        if face_photo is not None
                        else None
                    )

                if len(self.samples) == SAMPLES_REQUIRED:
                    saved_samples = [
                        sample.copy()
                        for sample in self.samples
                    ]
                    saved_photo = (
                        self.best_register_photo.copy()
                        if self.best_register_photo is not None
                        else None
                    )
                    try:
                        # Keep the proven local profile as the login source/fallback.
                        save_profile_once(
                            self.enroll_user,
                            saved_samples,
                        )
                        self.user_id = self.enroll_user
                        self.state = "registered"

                        # Cloud sync is intentionally asynchronous so network latency
                        # cannot block camera release / Pose handoff / formal re-login.
                        self._start_supabase_sync(
                            self.enroll_user,
                            saved_samples,
                            saved_photo,
                        )
                    except (OSError, ValueError) as exc:
                        self.last_error = str(exc)
                        self.state = "error"

                    self.samples = []
                    self.best_register_photo = None
                    self.best_register_photo_score = -1.0
                    self.stop_event.set()
            elif self.operation == "login":
                # 必須先真的進入框框，且從進框那一刻起完整等待 warm-up。
                # 這段時間 MediaPipe 舉手取消仍持續運作。
                if (
                    self.login_match_allowed_at <= 0.0
                    or monotonic() < self.login_match_allowed_at
                ):
                    self.match_name = None
                    self.match_streak = 0
                    return

                scores = {name: float(np.max(templates @ vector))
                          for name, templates in self.profiles.items()}
                ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                name, top_score = ordered[0]
                runner_up_name = ordered[1][0] if len(ordered) > 1 else None
                runner_up = ordered[1][1] if len(ordered) > 1 else -1.0
                score_margin = top_score - runner_up

                # Diagnostic only: do NOT change threshold or match behavior.
                # Log periodically so we can see whether failures come from
                # low similarity or an ambiguity-margin collision.
                if count == 1 or count % 30 == 0:
                    logging.info(
                        "Face match debug | best=%s | score=%.4f | "
                        "runner_up=%s | runner_up_score=%.4f | "
                        "margin=%.4f | threshold=%.4f | "
                        "required_margin=%.4f | streak=%d",
                        name,
                        top_score,
                        runner_up_name if runner_up_name is not None else "-",
                        runner_up,
                        score_margin,
                        MATCH_THRESHOLD,
                        AMBIGUITY_MARGIN,
                        self.match_streak,
                    )

                accepted = (top_score >= MATCH_THRESHOLD and
                            score_margin >= AMBIGUITY_MARGIN)
                if accepted:
                    if self.match_name == name:
                        self.match_streak += 1
                    else:
                        self.match_name, self.match_streak = name, 1
                    if self.match_streak >= MATCHES_REQUIRED:
                        self.user_id = name
                        self.authenticated = True
                        self.state = "matched_experimental"
                        self.stop_event.set()
                else:
                    self.match_name, self.match_streak = None, 0

    def _camera_worker(self, stop_event):
        camera = None
        try:
            if stop_event.is_set():
                return
            camera = Main()
            camera.stop_event = stop_event
            camera.on_embedding = self._process_embedding
            camera.on_face_state = self._process_face_state
            camera.on_cancel_gesture = self._process_cancel_gesture
            with self.lock:
                if not stop_event.is_set():
                    self.state = "waiting_face"
            # Stops also on timeout, including when no face is detected.
            timer = threading.Timer(OPERATION_TIMEOUT, stop_event.set)
            timer.daemon = True
            timer.start()
            try:
                camera.run(stop_event)
            finally:
                timer.cancel()
            with self.lock:
                if not stop_event.is_set() or self.state == "running":
                    self.state = "timeout"
        except Exception as exc:
            logging.exception("Face camera worker failed")
            with self.lock:
                self.last_error = str(exc)
                self.authenticated = False
                self.user_id = None
                self.state = "error"
        finally:
            # If camera.run() was entered, Main.run() closes self.device in its own
            # finally clause before control reaches here. If Main() failed during
            # construction, DepthAI.__init__ also closes any partially opened device.
            # Therefore camera_released=True here is the handoff-safe signal for Unity.
            with self.lock:
                if self.state == "stopping":
                    self.state = self.stop_target
                elif self.state in (
                    "starting",
                    "waiting_face",
                    "multiple_faces",
                    "adjust_position",
                    "running",
                ):
                    self.state = (
                        "timeout"
                        if monotonic() >= self.deadline
                        else self.stop_target
                    )
                self.samples = []
                self.best_register_photo = None
                self.best_register_photo_score = -1.0
                self.operation = None
                self.profiles = {}
                self.camera_released = True

            logging.info("Face camera released; OAK-D available for handoff")

    def shutdown(self):
        with self.lock:
            self.stop_event.set()
            worker = self.worker
        if worker is not None and worker.is_alive():
            worker.join()  # Avoid releasing the process while the device is still open.
        with self.lock:
            self.camera_released = True


if __name__ == "__main__":
    if args.mode == "idle":
        logging.info("IDLE: no camera opened; service not started")
    elif args.mode == "login":
        parser.error("Manual camera mode disabled in biometric prototype; start --mode server and send explicit consent")
    elif args.mode == "server":
        if preview:
            logging.warning("--preview is for manual mode only; server mode runs headless")
            preview = False
        if _supabase_is_configured():
            logging.info(
                "Supabase biometric sync configured "
                "(secret key loaded from environment)."
            )
        else:
            logging.warning(
                "Supabase biometric sync is NOT configured; "
                "local .npz registration will still work."
            )

        controller = FaceController()
        try:
            run_server(controller.handle)
        except KeyboardInterrupt:
            logging.info("TCP server interrupted")
        finally:
            controller.shutdown()
