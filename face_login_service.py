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

import cv2
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

    def run_face_mn(self):
        nn_data = self.mfd_nn.tryGet()
        if nn_data is None:
            return False

        bboxes = nn_data.detections
        # Reject ambiguous/multiple-face frames for both enrollment and matching.
        if len(bboxes) != 1:
            return False
        for bbox in bboxes:
            if self.stop_event.is_set():
                break
            face_coord = frame_norm(
                self.frame.shape[:2], *[bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
            )
            crop = self.frame[face_coord[1]:face_coord[3], face_coord[0]:face_coord[2]]
            if crop.size == 0:
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
                    self.on_embedding(self.embedding_count, results)
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
        self.deadline = 0.0
        self.match_name = None
        self.match_streak = 0
        self.profiles = {}

    def status(self):
        with self.lock:
            result = {"ok": True, "state": self.state,
                      "embedding_count": self.embedding_count,
                      "authenticated": self.authenticated,
                      "user_id": self.user_id, "error": self.last_error}
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
                profiles = load_profiles()
                if not profiles:
                    return {"ok": False, "error": "No enrolled profiles; register with consent first"}
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
                self.deadline = monotonic() + OPERATION_TIMEOUT
                self.match_name = None
                self.match_streak = 0
                self.profiles = profiles if command == "login" else {}
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
                if self.state in ("starting", "running", "stopping") or (
                    self.worker is not None and self.worker.is_alive()
                ):
                    self.stop_target = target
                    self.stop_event.set()
                    self.state = "stopping"
                else:
                    self.state = target
                    self.last_error = None
                    self.operation = None
                state = self.state
            return {"ok": True, "state": state,
                    "message": "Wait for status=guest/idle before starting another mode"}
        return {"ok": False, "error": "Unknown command"}

    def _process_embedding(self, count, raw):
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
                if len(self.samples) == SAMPLES_REQUIRED:
                    try:
                        # No face images; no overwrite of existing profile.
                        save_profile_once(self.enroll_user, self.samples)
                        self.user_id = self.enroll_user
                        self.state = "registered"
                    except (OSError, ValueError) as exc:
                        self.last_error = str(exc)
                        self.state = "error"
                    self.samples = []
                    self.stop_event.set()
            elif self.operation == "login":
                scores = {name: float(np.max(templates @ vector))
                          for name, templates in self.profiles.items()}
                ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                name, top_score = ordered[0]
                runner_up = ordered[1][1] if len(ordered) > 1 else -1.0
                accepted = (top_score >= MATCH_THRESHOLD and
                            top_score - runner_up >= AMBIGUITY_MARGIN)
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
            with self.lock:
                if not stop_event.is_set():
                    self.state = "running"
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
            with self.lock:
                if self.state == "stopping":
                    self.state = self.stop_target
                elif self.state in ("starting", "running"):
                    self.state = "timeout" if monotonic() >= self.deadline else self.stop_target
                self.samples = []
                self.operation = None
                self.profiles = {}
            # Main.run() closes the OAK-D in its finally clause.

    def shutdown(self):
        with self.lock:
            self.stop_event.set()
            worker = self.worker
        if worker is not None and worker.is_alive():
            worker.join()  # Avoid releasing the process while the device is still open.


if __name__ == "__main__":
    if args.mode == "idle":
        logging.info("IDLE: no camera opened; service not started")
    elif args.mode == "login":
        parser.error("Manual camera mode disabled in biometric prototype; start --mode server and send explicit consent")
    elif args.mode == "server":
        if preview:
            logging.warning("--preview is for manual mode only; server mode runs headless")
            preview = False
        controller = FaceController()
        try:
            run_server(controller.handle)
        except KeyboardInterrupt:
            logging.info("TCP server interrupted")
        finally:
            controller.shutdown()
