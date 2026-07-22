__all__ = ['custom_formatwarning', 'ForensicFace', 'FaceResult', 'ModelSpec']
import onnxruntime
import cv2
import numpy as np
import os.path as osp
import warnings
from .backends import FaceBackend, create_backend
from .aligned_processing import AlignedFaceRunner
from .component_catalog import (
    DEFAULT_ALIASES,
    build_detector,
    build_embedding_estimators,
    build_face_estimator,
    build_quality_estimator,
    normalize_embedding_selection,
    selection_alias,
)
from .components import DEFAULT, ComponentBackend, ModelSpec
from .batch import process_images_batch as process_images_batch_workflow
from .comparison import (
    aggregate_from_images as aggregate_from_images_workflow,
    compare_faces,
)
from .utils import (
    DEFAULT_KEYPOINT_COLORS,
    aggregate_embeddings,
    freeze_env,
    transform_keypoints,
    annotate_img_with_kps,
)
from .ort_runtime_setup import configure_onnxruntime_acceleration
from .runtime_summary import print_initialization_summary
from .geometry import select_best_face
from .model_store import resolve_quality_model, resolve_recognition_model
from .mosaic import build_mosaic_from_aligned_faces, build_mosaic_from_images
from .recognition import RecognitionRunner
from .quality import QualityRunner
from .results import (
    FaceResult,
    build_align_result,
    build_embedding_result,
    build_face_result,
)
from .video import extract_faces_from_video


def custom_formatwarning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = custom_formatwarning

class ForensicFace:
    """
    Class for processing facial images to extract useful features for forensic analysis.
    """

    IMG_SIZE = (112, 112)
    KEYPOINT_RECOGNITION_MODELS = {"sepaelv6"}
    SEPAELV6_IMAGE_INPUT = "input_images"
    SEPAELV6_KEYPOINTS_INPUT = "keypoints"

    def __init__(
        self,
        models: list[str] | object = DEFAULT,
        model: str = None,
        det_size: int | object = DEFAULT,
        use_gpu: bool = True,
        gpu: int = 0,  # which GPU to use
        concat_embeddings: bool = True,
        extended=True,
        det_thresh: float | object = DEFAULT,
        backend_name: str = "onnx",
        backend: FaceBackend | None = None,
        models_root: str = osp.join(osp.expanduser("~"), ".forensicface", "models"),
        *,
        detection=DEFAULT,
        pose=DEFAULT,
        gender=DEFAULT,
        age=DEFAULT,
        quality=DEFAULT,
        embedding=DEFAULT,
    ):
        """
        A face comparison tool for forensic analysis and comparison of facial images.

        Args:
        - models (list[str]): The names of the face recognition models to use (default: ["sepaelv2"]).
        - model (str): [deprecated] The name of the face recognition model to use
        - det_size (int): The size of the input images for face detection (default: 320).
        - use_gpu (bool): Whether to use a GPU for inference (default: True).
        - gpu (int): The ID of the GPU to use (default: 0).
        - concat_embeddings (bool): If True, concatenates the embeddings of each model.
        - extended (bool): Whether to use extended modules (detection, landmark_3d_68, genderage) (default: True).
        - det_thresh (float): threshold for the face detector (default = 0.5).
        - detection, pose, gender, age, quality: keyword-only task selectors.
          Each accepts a built-in alias, ModelSpec, or constructed component;
          optional tasks also accept None to disable them.
        - embedding: keyword-only embedding selector accepting one selection,
          multiple selections, or None. It is mutually exclusive with models.
        """

        selector_values = {
            "detection": detection,
            "pose": pose,
            "gender": gender,
            "age": age,
            "quality": quality,
            "embedding": embedding,
        }
        explicit_selectors = {
            task for task, value in selector_values.items() if value is not DEFAULT
        }
        if explicit_selectors and models is not DEFAULT:
            raise ValueError(
                "'models' cannot be combined with explicit task selectors "
                f"{sorted(explicit_selectors)}. Use 'embedding' instead."
            )
        if explicit_selectors and model is not None:
            raise ValueError(
                "Deprecated 'model' cannot be combined with explicit task selectors. "
                "Use 'embedding' instead."
            )
        if explicit_selectors and backend is not None:
            raise ValueError(
                "'backend' cannot be combined with explicit task selectors. "
                "Inject individual task components instead."
            )
        if explicit_selectors and backend_name != "onnx":
            raise ValueError(
                "A non-default 'backend_name' cannot be combined with explicit "
                "task selectors. Inject individual task components instead."
            )
        if detection is None:
            raise ValueError("detection=None is not supported; detection is required.")

        det_size_was_explicit = det_size is not DEFAULT
        det_thresh_was_explicit = det_thresh is not DEFAULT
        det_size = 320 if det_size is DEFAULT else det_size
        det_thresh = 0.5 if det_thresh is DEFAULT else det_thresh

        if use_gpu:
            self.providers = configure_onnxruntime_acceleration(verbose=False)
        else:
            self.providers = ['CPUExecutionProvider']

        if model is not None:
            warnings.warn(
                "__init__: The 'model' parameter is deprecated and will be removed in a future release.\n"
                "Please use the 'models' parameter instead: models = ['model_name']",
                DeprecationWarning,
            )
            self.models = [model]
        else:
            self.models = ["sepaelv2"] if models is DEFAULT else list(models)
        self.models_root = models_root
        # backward compatibility with older versions in which models_root was os.expanduser("~/.insightface/models")
        if not osp.isdir(models_root):
            warnings.warn(
                f"Model root directory '{models_root}' does not exist. "
                f"Falling back to '{osp.join(osp.expanduser('~'), '.insightface', 'models')}'. "
                f"Please ensure that the models are downloaded in the correct directory.",
                Warning,
            )
            self.models_root = osp.join(osp.expanduser("~"), ".insightface", "models")
        self.det_size = (det_size, det_size)
        self.det_thresh = det_thresh
        self.extended = extended
        self.embedding_estimators = None
        self.quality_estimator = None

        if not explicit_selectors:
            self._initialize_legacy_pipeline(
                backend=backend,
                backend_name=backend_name,
                gpu=gpu,
                use_gpu=use_gpu,
            )
        else:
            self._initialize_component_pipeline(
                selector_values=selector_values,
                gpu=gpu,
                use_gpu=use_gpu,
                det_size_was_explicit=det_size_was_explicit,
                det_thresh_was_explicit=det_thresh_was_explicit,
            )

        self.environment = freeze_env()
        self.concat_embeddings = concat_embeddings
        print_initialization_summary(self)

    def _initialize_legacy_pipeline(self, *, backend, backend_name, gpu, use_gpu):
        """Initialize the unchanged pre-selector path for API compatibility."""
        self.rec_inference_sessions = [
            self._load_model(model_name, [self.providers[0]], gpu, self.models_root)
            for model_name in self.models
        ]
        if self.extended:
            allowed_modules = ["detection", "landmark_3d_68", "genderage"]
            self.ort_fiqa = onnxruntime.InferenceSession(
                resolve_quality_model(self.models_root, self.models[0]),
                providers=[self.providers[0]],
            )
            self.enabled_tasks = frozenset(
                {"detection", "pose", "gender", "age", "quality", "embedding"}
            )
        else:
            allowed_modules = ["detection"]
            self.ort_fiqa = None
            self.enabled_tasks = frozenset({"detection", "embedding"})

        self.backend = backend or create_backend(
            backend_name=backend_name,
            model_name=self.models[0],
            allowed_modules=allowed_modules,
            providers=[self.providers[0]],
            ctx_id=gpu if use_gpu else -1,
            det_size=self.det_size,
            det_thresh=self.det_thresh,
            models_root=self.models_root,
        )

    def _initialize_component_pipeline(
        self,
        *,
        selector_values,
        gpu,
        use_gpu,
        det_size_was_explicit,
        det_thresh_was_explicit,
    ):
        providers = [self.providers[0]]
        ctx_id = gpu if use_gpu else -1
        embedding_selection = selector_values["embedding"]
        if embedding_selection is DEFAULT:
            embedding_selection = DEFAULT_ALIASES["embedding"]
        legacy_model_names = []
        if embedding_selection is not None:
            for selection in normalize_embedding_selection(embedding_selection):
                alias = selection_alias(selection)
                if isinstance(alias, str):
                    legacy_model_names.append(alias)

        detection_selection = selector_values["detection"]
        if detection_selection is DEFAULT:
            detection_selection = DEFAULT_ALIASES["detection"]
        detector_alias = (
            detection_selection.alias
            if isinstance(detection_selection, ModelSpec)
            else detection_selection
        )
        if detector_alias != "scrfd" and (
            det_size_was_explicit or det_thresh_was_explicit
        ):
            raise ValueError(
                "Legacy 'det_size' and 'det_thresh' apply only to SCRFD. "
                "Configure another detector through ModelSpec options."
            )
        detector = build_detector(
            detection_selection,
            models_root=self.models_root,
            providers=providers,
            ctx_id=ctx_id,
            det_size=self.det_size,
            det_thresh=self.det_thresh,
            legacy_model_names=legacy_model_names,
        )
        effective_input_size = getattr(detector, "input_size", None)
        if effective_input_size is not None:
            self.det_size = tuple(effective_input_size)

        if embedding_selection is None:
            self.embedding_estimators = []
        else:
            self.embedding_estimators = build_embedding_estimators(
                embedding_selection,
                models_root=self.models_root,
                providers=providers,
            )
        self.models = [
            estimator.name for estimator in self.embedding_estimators
        ]
        self.rec_inference_sessions = [
            getattr(estimator, "session", None)
            for estimator in self.embedding_estimators
        ]

        optional_selections = {}
        for task in ("pose", "gender", "age", "quality"):
            value = selector_values[task]
            optional_selections[task] = (
                DEFAULT_ALIASES[task] if value is DEFAULT and self.extended else
                None if value is DEFAULT else value
            )

        estimators = []
        estimator_cache = {}
        for task in ("pose", "gender", "age"):
            selection = optional_selections[task]
            if selection is None:
                continue
            cache_key = self._estimator_cache_key(selection)
            estimator = estimator_cache.get(cache_key)
            if estimator is None:
                estimator = build_face_estimator(
                    task,
                    selection,
                    models_root=self.models_root,
                    providers=providers,
                    ctx_id=ctx_id,
                    legacy_model_names=legacy_model_names,
                )
                estimator_cache[cache_key] = estimator
            elif task not in estimator.capabilities:
                estimator = build_face_estimator(
                    task,
                    selection,
                    models_root=self.models_root,
                    providers=providers,
                    ctx_id=ctx_id,
                    legacy_model_names=legacy_model_names,
                )
            estimators.append(estimator)

        quality_selection = optional_selections["quality"]
        if quality_selection is not None:
            legacy_name = self.models[0] if self.models else DEFAULT_ALIASES["embedding"]
            self.quality_estimator = build_quality_estimator(
                quality_selection,
                models_root=self.models_root,
                legacy_model_name=legacy_name,
                providers=providers,
            )
        self.ort_fiqa = getattr(self.quality_estimator, "session", None)
        self.enabled_tasks = frozenset(
            {"detection"}
            | ({"embedding"} if self.embedding_estimators else set())
            | {task for task, value in optional_selections.items() if value is not None}
        )
        self.backend = ComponentBackend(
            detector=detector,
            estimators=estimators,
            enabled_tasks=set(self.enabled_tasks),
        )

    @staticmethod
    def _estimator_cache_key(selection):
        if isinstance(selection, str):
            return ("alias", selection)
        if isinstance(selection, ModelSpec):
            if selection.path is None and not selection.options:
                return ("alias", selection.alias)
            return (
                "spec",
                selection.alias,
                str(selection.path),
                tuple(
                    sorted((key, repr(value)) for key, value in selection.options.items())
                ),
            )
        return ("object", id(selection))

    def _get_loaded_modules(self) -> list[str]:
        modules = ["detection"]
        if getattr(self.backend, "landmark_model", None) is not None:
            modules.append("headpose")
        if getattr(self.backend, "genderage_model", None) is not None:
            modules.append("genderage")
        if "quality" in getattr(self, "enabled_tasks", set()):
            modules.append("cr_fiqa")
        
        return modules

    def _load_model(self, model_name, providers, gpu, models_root):
        """Loads a single ONNX face recognition model.

        Tries the new shared layout first
        (``<models_root>/recognition/<model_name>/*face*.onnx``)
        and falls back to the legacy per-model layout
        (``<models_root>/<model_name>/*/*face*.onnx``).
        """
        return onnxruntime.InferenceSession(
            resolve_recognition_model(models_root, model_name),
            providers=providers,
        )

    def _recognition_runner(self):
        return RecognitionRunner(
            models=self.models,
            rec_inference_sessions=self.rec_inference_sessions,
            concat_embeddings=self.concat_embeddings,
            image_size=self.IMG_SIZE,
            keypoint_recognition_models=self.KEYPOINT_RECOGNITION_MODELS,
            image_input_name=self.SEPAELV6_IMAGE_INPUT,
            keypoints_input_name=self.SEPAELV6_KEYPOINTS_INPUT,
            embedding_estimators=self.embedding_estimators,
        )

    def _quality_runner(self):
        return QualityRunner(
            estimator=self.quality_estimator,
            legacy_session=(
                None if self.quality_estimator is not None
                else getattr(self, "ort_fiqa", None)
            ),
            image_size=self.IMG_SIZE,
        )

    def _aligned_face_runner(self):
        return AlignedFaceRunner(
            recognition=self._recognition_runner(),
            quality=self._quality_runner(),
        )

    def process_image_single_face(
        self,
        imgpath: str,
        draw_keypoints=False,
        keypoint_colors: tuple[str, str, str, str, str] = DEFAULT_KEYPOINT_COLORS,
    ):  # Path to image to be processed
        """
        Process a an image considering it has a single face and extract useful features for forensic analysis.
        If more than one face is detected, the largest face will be returned.
        THIS METHOD IS DEPRECATED AND WILL BE REMOVED IN A FUTURE RELEASE. Use process_image instead.
        """
        warnings.warn(
            "process_image_single_face: This method is deprecated and will be removed in a future release.\n"
            "Use the 'process_image' method instead. ",
            DeprecationWarning,
        )
        bgr_img = self._load_image(imgpath)
        return self.process_image(
            bgr_img,
            draw_keypoints=draw_keypoints,
            keypoint_colors=keypoint_colors,
            single_face=True,
            select_single_face_by="size",
        )

    def detect_and_align(
        self,
        imgpath: str | np.ndarray,
        *,
        single_face: bool = True,
        select_single_face_by: str = "size",
    ) -> dict | list[dict] | None:
        """Detect faces and return aligned crops without embedding/FIQA.

        Useful in two scenarios:
        1. Batched extraction: run ``detect_and_align`` per image to fill a
           buffer of aligned crops, then call ``process_aligned_faces_batch``
           once per chunk. Lets ONNX use real batch parallelism on GPU.
        2. Materialize aligned crops on disk for later re-extraction
           with newer recognition models.

        Args:
            imgpath: path to the image (str) or a BGR ``np.ndarray``.
            single_face: when True, return only the best detected face.
            select_single_face_by: ``"size"`` or ``"centrality"``;
                applied only when single_face=True and multiple faces
                are detected.

        Returns:
            dict | list[dict] | None: With ``single_face=True``, returns
            one aligned-face dictionary or ``None`` when no face is detected.
            With ``single_face=False``, returns a list of dictionaries,
            possibly empty. Each dictionary includes ``aligned_face`` (RGB
            ``ndarray``), ``bbox``, ``keypoints``, ``aligned_keypoints``,
            and ``det_score``. When ``extended=True``, it also includes
            ``gender``, ``age``, and ``pose``.
        """
        bgr_img = self._load_image(imgpath)
        faces = self.backend.detect_faces(bgr_img)
        if len(faces) == 0:
            return None if single_face else []

        if single_face:
            faces = [
                select_best_face(
                    bgr_img.shape,
                    faces,
                    criterion=select_single_face_by,
                )
            ]

        results = []
        for face in faces:
            aligned_bgr = (
                face.aligned_bgr
                if face.aligned_bgr is not None
                else self.backend.norm_crop(bgr_img, face.kps)
            )
            aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
            # `aligned_keypoints` are in the 112×112 coordinate system of
            # the aligned face — this is what KEYPOINT_RECOGNITION_MODELS
            # (sepaelv6/KPRPE) consume as a second input. They are
            # precomputed here so `process_images_batch` can stack them
            # without having to run `_align_keypoints` again.
            aligned_kps = (
                face.aligned_keypoints
                if face.aligned_keypoints is not None
                else self._align_keypoints(face.kps)
            )
            item = build_align_result(
                aligned_face=aligned_rgb,
                bbox=face.bbox,
                keypoints=face.kps,
                aligned_keypoints=aligned_kps,
                det_score=face.det_score,
                extended=self.extended,
                gender=face.gender,
                age=face.age,
                pose=face.pose,
                enabled_tasks=self.enabled_tasks,
            )
            results.append(item)

        return results[0] if single_face else results

    def process_image(
        self,
        imgpath,
        single_face=True,
        draw_keypoints=False,
        keypoint_colors: tuple[str, str, str, str, str] = DEFAULT_KEYPOINT_COLORS,
        select_single_face_by="size",
    ):
        """Process an image assuming one or multiple faces.
        Args:
            - imgpath (str | np.ndarray): Path to the input image or cv2 image array in BGR.
            - draw_keypoints (bool): If set to True, draw the keypoints on the aligned face.
            - keypoint_colors (tuple[str, str, str, str, str]): colors used for
                the five aligned keypoints when draw_keypoints=True.
            - single_face (bool): If set to True, process only one face in the image.
            - select_single_face_by (str): criterion to select the face in the image, if more than one face is detected.
                Only applicable when single_face == True. Must be either 'size' or 'centrality'.
        Returns:
            If single_face==True, return a dictionary containing the following keys:
                - 'keypoints': A 2D numpy array of shape (5, 2) containing the facial keypoints
                        for each face in the image. The keypoints are ordered as follows:
                       left eye, right eye, nose tip, left mouth corner, and right mouth corner.

                - 'ipd': A float representing the inter-pupillary distance for each face in the image.

                - 'embedding': A 1D numpy array of shape (512,) containing the facial embedding
                       for each face in the image.
                       If the concat_emmbeddings == True, keys for each model are used with the names <model_name>_embedding

                - 'bbox': A 1D numpy array of shape (4,) containing the bounding box coordinates for each face
                  in the image. The coordinates are ordered as follows: (xmin, ymin, xmax, ymax).

                - 'aligned_face': A 3D numpy array of shape (H, W, C) in RGB order containing the aligned face image for
                          each face in the image. The image has been cropped and aligned based on the
                          facial keypoints.

                - 'det_score': A float representing the face detection score.

                If 'extended' is set to True, the dictionary will also contain the following keys:
                - 'gender': Estimated sex for each face in the image. Possible values: 'M' (male) and 'F' (female).

                - 'age': An integer representing the estimated age for each face in the image.

                - 'pitch': A float representing the pitch angle for each face in the image.

                - 'yaw': A float representing the yaw angle for each face in the image.

                - 'roll': A float representing the roll angle for each face in the image.

                - 'fiqa_score': A float indicating facial image quality.
            
            If single_face==False, return a list of dictionaries, each containing the same keys as described above for each detected face in the image.
        """
        if single_face == True:
            warnings.warn(
                "process_image: The return of this function when 'single_face = True' will change in a future release.\n"
                "Instead of returning a dict, it will return a list (with one dict). ",
                FutureWarning,
            )
        bgr_img = self._load_image(imgpath)
        faces = self.backend.detect_faces(bgr_img)
        if len(faces) == 0:
            return []

        if single_face:
            faces = [
                select_best_face(
                    bgr_img.shape,
                    faces,
                    criterion=select_single_face_by,
                )
            ]

        results = []
        for face in faces:
            bgr_aligned_face = (
                face.aligned_bgr
                if face.aligned_bgr is not None
                else self.backend.norm_crop(bgr_img, face.kps)
            )
            aligned_kps = (
                face.aligned_keypoints
                if face.aligned_keypoints is not None
                else self._align_keypoints(face.kps)
            )
            embeddings, fiqa_score = self._aligned_face_runner().compute_one(
                bgr_aligned_face, aligned_keypoints=aligned_kps
            )
            if draw_keypoints:
                bgr_aligned_face = self._draw_keypoints_on_aligned_face(
                    bgr_aligned_face, aligned_kps, keypoint_colors=keypoint_colors
                )
            result = build_face_result(
                aligned_face=cv2.cvtColor(bgr_aligned_face, cv2.COLOR_BGR2RGB),
                bbox=face.bbox,
                keypoints=face.kps,
                aligned_keypoints=aligned_kps,
                det_score=face.det_score,
                embeddings=embeddings,
                fiqa_score=fiqa_score,
                models=self.models,
                extended=self.extended,
                concat_embeddings=self.concat_embeddings,
                gender=face.gender,
                age=face.age,
                pose=face.pose,
                enabled_tasks=self.enabled_tasks,
            )
            results.append(result)

        return results if not single_face else results[0]

    def _align_keypoints(self, keypoints):
        M = self.backend.estimate_norm(keypoints)
        return transform_keypoints(keypoints=keypoints, M=M)

    def _draw_keypoints_on_aligned_face(
        self,
        bgr_aligned_face,
        aligned_keypoints,
        keypoint_colors=DEFAULT_KEYPOINT_COLORS,
    ):
        aligned_face = bgr_aligned_face.copy()
        annotated_aligned_face = annotate_img_with_kps(
            aligned_face, kps=aligned_keypoints, colors=keypoint_colors
        )
        return annotated_aligned_face

    def process_images_batch(
        self,
        imgpaths: list[str | np.ndarray],
        *,
        single_face: bool = True,
        select_single_face_by: str = "size",
        batch_size: int = 16,
    ) -> list:
        """Batched counterpart of ``process_image``.

        Pipeline:
        1. Per-image: ``detect_and_align``, accumulate aligned
           crops into a buffer.
        2. Per-chunk of ``batch_size``: recognition/FIQA inference in one
           ONNX call per loaded model.
        3. Reassemble into ``process_image``-compatible dicts.

        Args:
            imgpaths: iterable of image paths (str) or BGR ndarrays.
            single_face: when True, returns one dict per image (the
                best face) or ``None`` when no face is detected.
            select_single_face_by: ``"size"`` or ``"centrality"`` —
                only used with single_face=True.
            batch_size: number of faces fed to the recognition ONNX
                session at once. Default ``16`` is conservative — fits
                a single recognition model on an 8GB GPU (e.g. RTX 3070)
                with room for a second model. Raise it on bigger GPUs
                for more throughput; lower it on CPU. If the batch
                causes a CUDA OOM, the call auto-halves the batch and
                may warn while reducing the batch size.

        Returns:
            list: Results parallel to ``imgpaths``. With ``single_face=True``,
            each item is a ``dict`` compatible with ``process_image`` or
            ``None`` when no face is detected. With ``single_face=False``,
            each item is a list of per-image face dictionaries.
        """
        return process_images_batch_workflow(
            self,
            imgpaths,
            single_face=single_face,
            select_single_face_by=select_single_face_by,
            batch_size=batch_size,
        )

    def _load_image(self, imgpath):
        """Load image from file path or return the array if already loaded."""
        return cv2.imread(imgpath) if isinstance(imgpath, str) else imgpath.copy()

    def process_image_multiple_faces(
        self,
        imgpath: str,
        draw_keypoints=False,
        keypoint_colors: tuple[str, str, str, str, str] = DEFAULT_KEYPOINT_COLORS,
    ):
        """
        Process an image assuming multiple faces.
        THIS METHOD IS DEPRECATED AND WILL BE REMOVED IN A FUTURE RELEASE. Use process_image instead.
        """
        warnings.warn(
            "process_image_multiple_faces: This method is deprecated and will be removed in a future release.\n"
            "Use the 'process_image' method instead.",
            DeprecationWarning,
        )
        bgr_img = self._load_image(imgpath)
        return self.process_image(
            bgr_img,
            draw_keypoints=draw_keypoints,
            keypoint_colors=keypoint_colors,
            single_face=False,
        )

    def build_mosaic(
        self,
        img_path_list: list[str | np.ndarray],
        mosaic_shape: tuple[int, int],
        border: float = 0.03,
        save_to: str | None = None,
        draw_keypoints: bool = False,
        keypoint_colors: tuple[str, str, str, str, str] = DEFAULT_KEYPOINT_COLORS,
    ) -> np.ndarray:
        """
        Detect, align, and build a rectangular mosaic from original images.

        Args:
            img_path_list: list of paths to image files or list of bgr_images
            mosaic_shape: tuple of integers, (n_cols, n_rows)
            border: float, percent of image to use as white border
            save_to: optional path used to save the mosaic image.
            draw_keypoints: if True, draw keypoints on each aligned face.
            keypoint_colors: colors used for the five aligned keypoints when
                draw_keypoints=True.

        Returns:
            np.ndarray: OpenCV BGR image with mosaic.
        """
        return build_mosaic_from_images(
            self,
            img_path_list,
            mosaic_shape,
            border=border,
            save_to=save_to,
            draw_keypoints=draw_keypoints,
            keypoint_colors=keypoint_colors,
        )

    def build_mosaic_from_aligned_faces(
        self,
        aligned_faces: list[np.ndarray] | np.ndarray,
        mosaic_shape: tuple[int, int],
        border: float = 0.03,
        save_to: str | None = None,
        draw_keypoints: bool = False,
        keypoints: list[np.ndarray] | np.ndarray | None = None,
        keypoint_colors: tuple[str, str, str, str, str] = DEFAULT_KEYPOINT_COLORS,
    ) -> np.ndarray:
        """
        Build a rectangular mosaic from already aligned RGB face images.

        This is a convenience wrapper around
        ``forensicface.mosaic.build_mosaic_from_aligned_faces`` using this
        instance's configured face image size.

        Args:
            aligned_faces: list/array of already aligned RGB face images.
            mosaic_shape: tuple of integers, (n_cols, n_rows)
            border: float, percent of image to use as white border
            save_to: optional path used to save the mosaic image.
            draw_keypoints: if True, draw keypoints on each aligned face.
            keypoints: optional list/array of aligned keypoints, one per aligned face.
                Required when draw_keypoints=True.
            keypoint_colors: colors used for the five aligned keypoints when
                draw_keypoints=True.

        Returns:
            np.ndarray: OpenCV BGR image with mosaic.
        """
        return build_mosaic_from_aligned_faces(
            aligned_faces,
            mosaic_shape,
            border=border,
            save_to=save_to,
            draw_keypoints=draw_keypoints,
            keypoints=keypoints,
            keypoint_colors=keypoint_colors,
            image_size=self.IMG_SIZE,
        )

    def compare(self, img1path: str, img2path: str) -> float:
        """
        Compute the similarity cosine between the embeddings of two face images.

        Args:
            img1path: Path to the first image file.
            img2path: Path to the second image file.

        Returns:
            float: Cosine similarity.
            The score ranges from -1.0 (most dissimilar) to 1.0 (most similar).

        Raises:
            ValueError: If ``concat_embeddings`` is False, because this method
                requires a single concatenated embedding for each image.
        """
        return compare_faces(self, img1path, img2path)

    def _require_embedding_models(self, api_name: str) -> None:
        if "embedding" not in self.enabled_tasks:
            raise ValueError(
                f"{api_name} requires an embedding model; initialize "
                "ForensicFace with embedding=... instead of embedding=None."
            )

    def aggregate_embeddings(
        self,
        embeddings: np.ndarray,
        weights: np.ndarray | None = None,
        method: str = "mean",
    ) -> np.ndarray:
        """
        Aggregates multiple embeddings into a single embedding.

        Args:
            embeddings: A 2D array of shape (num_embeddings, embedding_dim) containing the embeddings to be
                aggregated.
            weights: A 1D array of shape (num_embeddings,) containing the weights to be assigned
                to each embedding. If not provided, all embeddings are equally weighted.

            method: choice of agregating based on the mean or median of the embeddings. Possible values are
                'mean' and 'median'.

        Returns:
            np.ndarray: A 1D array of shape (embedding_dim,) containing the aggregated embedding.
        """
        return aggregate_embeddings(embeddings, weights=weights, method=method)

    def aggregate_from_images(
        self,
        list_of_image_paths: list[str],
        method: str = "mean",
        quality_weight: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray] | list:
        """
        Given a list of image paths, this method returns the average embedding of all faces found in the images.

        Args:
            list_of_image_paths: List of paths to images.
            method: choice of agregating based on the mean or median of the embeddings. Possible values are
                'mean' and 'median'.
            quality_weight: If True, use the FIQA(L) score as a weight for aggregation.

        Returns:
            np.ndarray | dict[str, np.ndarray] | list: If one or more faces
            are found and ``concat_embeddings=True``, returns a 1D numpy array
            representing the average embedding. If ``concat_embeddings=False``,
            returns a dictionary with one aggregated embedding per model using
            keys in the form ``embedding_<model_name>``. If no faces are found,
            returns an empty list.
        """
        return aggregate_from_images_workflow(
            self,
            list_of_image_paths,
            method=method,
            quality_weight=quality_weight,
        )

    def extract_faces(
        self,
        video_path: str,  # path to video file
        dest_folder: str = None,  # folder used to save extracted faces. If not provided, a new folder with the video name is created
        every_n_frames: int = 1,  # skip some frames
        margin: float = 2.0,  # margin to add to each face, w.r.t. detected bounding box
        start_from: float = 0.0,  # seconds after video start to begin processing
        export_metadata: bool = False,  # if True, export facial keypoints, bounding box, ipd, fiqa_score, pitch, yaw, roll, and embedding
    ) -> int:
        """
        Extracts faces from a video and saves them as individual images.

        Args:
            video_path: The path to the input video file.
            dest_folder: The path to the output folder. If not provided, a
                new folder with the same name as the input video file is created.
            every_n_frames: Extract faces from every n-th frame.
            margin: The factor by which the detected face bounding box should
                be extended.
            start_from: The time point, in seconds, after which the video
                frames should be processed.
            export_metadata: If True, export facial metadata for each face.

        Returns:
            int: The number of extracted faces.
        """
        return extract_faces_from_video(
            self,
            video_path,
            dest_folder=dest_folder,
            every_n_frames=every_n_frames,
            margin=margin,
            start_from=start_from,
            export_metadata=export_metadata,
        )

    def process_aligned_face_image(
        self, rgb_aligned_face: np.ndarray, keypoints: np.ndarray | None = None
    ):
        """
        Process an already aligned RGB face image.

        Args:
            rgb_aligned_face: RGB face image with shape (112, 112, 3).
            keypoints: Optional 5x2 keypoints in the aligned 112x112 image
                coordinate system. Required when a keypoint-aware recognition
                model such as sepaelv6 is loaded.
        """
        self._require_embedding_models("process_aligned_face_image()")
        if rgb_aligned_face.shape != (*self.IMG_SIZE, 3):
            raise ValueError(
                f"rgb_aligned_face must have shape {(*self.IMG_SIZE, 3)}; "
                f"got {rgb_aligned_face.shape}."
            )
        bgr_aligned_face = rgb_aligned_face[..., ::-1].copy()
        embeddings, fiqa_score = self._aligned_face_runner().compute_one(
            bgr_aligned_face, aligned_keypoints=keypoints
        )

        return build_embedding_result(
            embeddings=embeddings,
            fiqa_score=fiqa_score,
            models=self.models,
            extended=self.extended,
            concat_embeddings=self.concat_embeddings,
            enabled_tasks=self.enabled_tasks,
        )

    def process_aligned_faces_batch(
        self,
        rgb_aligned_faces: np.ndarray,
        aligned_keypoints_batch: np.ndarray | None = None,
    ) -> list[FaceResult]:
        """
        Process already aligned RGB face images in one recognition batch.

        Args:
            rgb_aligned_faces: RGB aligned face images with shape
                ``(N, 112, 112, 3)``.
            aligned_keypoints_batch: Optional keypoints with shape
                ``(N, 5, 2)`` in the aligned 112x112 image coordinate system.
                Required when a keypoint-aware recognition model such as
                sepaelv6 is loaded.

        Returns:
            list[FaceResult]: One result per aligned face. Each result includes
            ``embedding`` when ``concat_embeddings=True`` or one
            ``embedding_<model_name>`` key per model otherwise. When
            ``extended=True``, each result also includes ``fiqa_score``.
        """
        self._require_embedding_models("process_aligned_faces_batch()")
        rgb_aligned_faces = np.asarray(rgb_aligned_faces)
        if (
            rgb_aligned_faces.ndim != 4
            or rgb_aligned_faces.shape[1:] != (*self.IMG_SIZE, 3)
        ):
            raise ValueError(
                f"rgb_aligned_faces must have shape (N, {self.IMG_SIZE[0]}, "
                f"{self.IMG_SIZE[1]}, 3); got {rgb_aligned_faces.shape}."
            )

        bgr_aligned_faces = rgb_aligned_faces[..., ::-1].copy()
        embeddings, fiqa_scores = self._aligned_face_runner().compute_batch(
            bgr_aligned_faces,
            aligned_keypoints_batch=aligned_keypoints_batch,
        )

        return [
            build_embedding_result(
                embeddings=embeddings,
                fiqa_score=fiqa_scores,
                models=self.models,
                extended=self.extended,
                concat_embeddings=self.concat_embeddings,
                enabled_tasks=self.enabled_tasks,
                index=idx,
            )
            for idx in range(rgb_aligned_faces.shape[0])
        ]
