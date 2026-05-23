__all__ = ['custom_formatwarning', 'ForensicFace', 'FaceResult']
import onnxruntime
import cv2
import numpy as np
import os.path as osp
import warnings
from .backends import FaceData, FaceBackend, create_backend
from .utils import freeze_env, transform_keypoints, annotate_img_with_kps
from .ort_runtime_setup import configure_onnxruntime_acceleration
from .runtime_summary import print_initialization_summary
from .geometry import extend_bbox, select_best_face
from .model_store import resolve_quality_model, resolve_recognition_model
from .mosaic import build_aligned_face_mosaic
from .preprocessing import normalize_aligned_keypoints, to_ada_input
from .recognition import (
    RecognitionRunner,
    build_keypoint_model_inputs,
    looks_like_cuda_oom,
    try_compute_embeddings_batch,
)
from .results import (
    FaceResult,
    build_align_result,
    build_face_result,
    build_face_result_from_align_result,
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
        models: list[str] = ["sepaelv2"],
        model: str = None,
        det_size: int = 320,
        use_gpu: bool = True,
        gpu: int = 0,  # which GPU to use
        concat_embeddings: bool = True,
        extended=True,
        det_thresh: float = 0.5,
        backend_name: str = "onnx",
        backend: FaceBackend | None = None,
        models_root: str = osp.join(osp.expanduser("~"), ".forensicface", "models"),
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
        """

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
            self.models = models
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
        self.rec_inference_sessions = [
            self._load_model(model_name, [self.providers[0]], gpu, self.models_root) for model_name in self.models
        ]

        self.det_size = (det_size, det_size)
        self.det_thresh = det_thresh
        self.extended = extended
        if self.extended:
            allowed_modules = ["detection", "landmark_3d_68", "genderage"]

            self.ort_fiqa = onnxruntime.InferenceSession(
                self._resolve_quality_model(self.models[0]),
                providers=[self.providers[0]],
            )
        else:
            allowed_modules = ["detection"]

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

        self.environment = freeze_env()
        self.concat_embeddings = concat_embeddings
        print_initialization_summary(self)

    def _get_loaded_modules(self) -> list[str]:
        modules = ["detection"]
        if getattr(self.backend, "landmark_model", None) is not None:
            modules.append("headpose")
        if getattr(self.backend, "genderage_model", None) is not None:
            modules.append("genderage")
        if self.extended:
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

    def _resolve_quality_model(self, model_name: str) -> str:
        """Resolves the CR-FIQA quality model path.

        Tries the new shared layout first
        (``<models_root>/quality/cr_fiqa_l.onnx``) and falls back to the
        legacy per-model layout
        (``<models_root>/<model_name>/cr_fiqa/cr_fiqa_l.onnx``).
        """
        return resolve_quality_model(self.models_root, model_name)

    def _to_input_ada(self, aligned_bgr_img):
        """
        Preprocesses the input face(s) for the face recognition model.

        Args:
            aligned_bgr_img: Face image(s) in BGR order as a numpy array.
                Accepts a single image with shape ``(H, W, 3)`` or a batch
                with shape ``(N, H, W, 3)``.

        Returns:
            Preprocessed face image(s) as a numpy array with shape
            ``(1, 3, H, W)`` for a single input or ``(N, 3, H, W)`` for
            a batch input. Same normalization in both cases — single
            source of truth for image preprocessing.
        """
        return to_ada_input(aligned_bgr_img, image_size=self.IMG_SIZE)

    def _recognition_runner(self):
        return RecognitionRunner(
            models=self.models,
            rec_inference_sessions=self.rec_inference_sessions,
            ort_fiqa=getattr(self, "ort_fiqa", None),
            extended=self.extended,
            concat_embeddings=self.concat_embeddings,
            image_size=self.IMG_SIZE,
            keypoint_recognition_models=self.KEYPOINT_RECOGNITION_MODELS,
            image_input_name=self.SEPAELV6_IMAGE_INPUT,
            keypoints_input_name=self.SEPAELV6_KEYPOINTS_INPUT,
        )

    def _get_best_face(self, img, faces, criterion="size"):
        """Get the best face based on a criterion: 'centrality' or 'size'."""
        return select_best_face(img.shape, faces, criterion=criterion)

    def process_image_single_face(
        self, imgpath: str, draw_keypoints=False
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
            single_face=True,
            select_single_face_by="size",
        )

    def align_only(
        self,
        imgpath: str | np.ndarray,
        *,
        single_face: bool = True,
        select_single_face_by: str = "size",
    ) -> dict | list[dict] | None:
        """Detect + align without extracting embedding/FIQA.

        Useful in two scenarios:
        1. Batched extract: run align_only per-image to fill a buffer of
           aligned crops, then call ``_compute_embeddings_batch`` once
           per chunk. Lets ONNX use real batch parallelism on GPU.
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
                self._get_best_face(
                    bgr_img, faces, criterion=select_single_face_by
                )
            ]

        results = []
        for face in faces:
            aligned_bgr = self.backend.norm_crop(bgr_img, face.kps)
            aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
            # `aligned_keypoints` are in the 112×112 coordinate system of
            # the aligned face — this is what KEYPOINT_RECOGNITION_MODELS
            # (sepaelv6/KPRPE) consume as a second input. They are
            # precomputed here so `process_images_batch` can stack them
            # without having to run `_align_keypoints` again.
            aligned_kps = self._align_keypoints(face.kps)
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
            )
            results.append(item)

        return results[0] if single_face else results

    def process_image(
        self,
        imgpath,
        single_face=True,
        draw_keypoints=False,
        select_single_face_by="size",
    ):
        """Process an image assuming one or multiple faces.
        Args:
            - imgpath (str | np.ndarray): Path to the input image or cv2 image array in BGR.
            - draw_keypoints (bool): If set to True, draw the keypoints on the aligned face.
            - single_face (bool): If set to True, assume the image may contain more than one face.
            - select_single_face_by (str): criterion to select the face in the image, if more than one face is detected.
                Only applicable when single_face == True. Must be either 'size' or 'centrality'.
        Returns:
            A dictionary containing the following keys:
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

                If the 'extended' is set to True, the dictionary will also contain the following keys:
                - 'gender': A string representing the gender for each face in the image.
                               Possible values are 'M' for male and 'F' for female.

                - 'age': An integer representing the estimated age for each face in the image.

                - 'pitch': A float representing the pitch angle for each face in the image.

                - 'yaw': A float representing the yaw angle for each face in the image.

                - 'roll': A float representing the roll angle for each face in the image.

                - fiqa_score: A float indicating facial image quality.
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
                self._get_best_face(bgr_img, faces, criterion=select_single_face_by)
            ]

        results = []
        for face in faces:
            bgr_aligned_face = self.backend.norm_crop(bgr_img, face.kps)
            aligned_kps = self._align_keypoints(face.kps)
            embeddings, fiqa_score = self._compute_embeddings(
                bgr_aligned_face, aligned_keypoints=aligned_kps
            )
            if draw_keypoints:
                bgr_aligned_face = self._draw_keypoints_on_aligned_face(
                    bgr_aligned_face, aligned_kps
                )
            result = self._assemble_result(
                face, bgr_aligned_face, embeddings, fiqa_score
            )
            results.append(result)

        return results if not single_face else results[0]

    def _align_keypoints(self, keypoints):
        M = self.backend.estimate_norm(keypoints)
        return transform_keypoints(keypoints=keypoints, M=M)

    def _draw_keypoints_on_aligned_face(self, bgr_aligned_face, aligned_keypoints):
        aligned_face = bgr_aligned_face.copy()
        annotated_aligned_face = annotate_img_with_kps(
            aligned_face, kps=aligned_keypoints, color="green"
        )
        return annotated_aligned_face

    def _compute_embeddings(self, bgr_aligned_face, aligned_keypoints=None):
        """Computes embeddings and FIQA score for an aligned face."""
        return self._recognition_runner().compute_one(
            bgr_aligned_face,
            aligned_keypoints=aligned_keypoints,
        )

    def _build_keypoint_model_inputs(
        self, model_name, rec_ort, img_to_input, aligned_keypoints
    ):
        return build_keypoint_model_inputs(
            model_name=model_name,
            rec_ort=rec_ort,
            img_to_input=img_to_input,
            aligned_keypoints=aligned_keypoints,
            image_size=self.IMG_SIZE,
            image_input_name=self.SEPAELV6_IMAGE_INPUT,
            keypoints_input_name=self.SEPAELV6_KEYPOINTS_INPUT,
        )

    def _to_keypoints_input(self, aligned_keypoints, model_name):
        return normalize_aligned_keypoints(
            aligned_keypoints,
            model_name=model_name,
            image_size=self.IMG_SIZE,
        )

    def _compute_embeddings_batch(
        self,
        bgr_aligned_batch: np.ndarray,
        aligned_keypoints_batch: np.ndarray = None,
    ):
        """Computes embeddings + FIQA for ``N`` aligned faces in parallel.

        Batched counterpart of ``_compute_embeddings``. Each ONNX session
        is invoked once with input shape ``(N, 3, 112, 112)`` instead of
        ``N`` calls with shape ``(1, 3, 112, 112)``. Speedup scales with
        ``N`` until GPU memory or Python overhead dominates.

        Args:
            bgr_aligned_batch: ``ndarray (N, 112, 112, 3)`` BGR uint8 —
                aligned crops, typically the output of ``align_only``
                stacked along axis 0.
            aligned_keypoints_batch: ``ndarray (N, 5, 2)`` float — aligned
                5-point keypoints in the 112×112 coordinate system, one
                per face. Required when any loaded recognition model is
                in ``KEYPOINT_RECOGNITION_MODELS`` (e.g. sepaelv6/KPRPE);
                ignored otherwise.

        Returns:
            embeddings:
                If ``concat_embeddings=True``: ``ndarray (N, total_dim)``
                with embeddings concatenated across all loaded models.
                Otherwise: list of ndarrays, one per loaded model, each
                of shape ``(N, model_dim)``.
            fiqa_scores:
                ``ndarray (N,)`` with the FIQA score per face, or ``None``
                when ``extended=False``.

        Notes:
            Equivalent to calling ``_compute_embeddings`` N times:
            ONNX Runtime performs the same matmul ops, only the
            parallel layout changes. Results are equivalent for
            practical purposes (e.g. cosine similarity), but on GPU
            the parallel reductions are not bit-exact deterministic,
            so embeddings may differ by tiny amounts vs the per-image
            path.
        """
        return self._recognition_runner().compute_batch(
            bgr_aligned_batch,
            aligned_keypoints_batch=aligned_keypoints_batch,
        )

    @staticmethod
    def _looks_like_cuda_oom(exc: BaseException) -> bool:
        """Best-effort match for CUDA out-of-memory errors raised by
        ONNX Runtime. Provider-specific exception types vary by build,
        so we string-match on the message — broad enough to catch
        ORT's CUDA, TRT, and DML provider OOMs."""
        return looks_like_cuda_oom(exc)

    def _try_compute_embeddings_batch(
        self, bgr_aligned_batch, aligned_keypoints_batch=None,
    ):
        """Calls ``_compute_embeddings_batch`` with CUDA OOM auto-retry.

        On OOM, halves the batch and recurses on each half, concatenating
        results to match the original call signature. Emits a one-line
        warning so the user notices and lowers ``batch_size`` upstream.
        Re-raises if even ``batch_size=1`` OOMs (= genuine out-of-memory,
        not just over-eager batching).

        ``aligned_keypoints_batch`` é fatiado em paralelo com
        ``bgr_aligned_batch`` quando fornecido — necessário pra modelos
        em ``KEYPOINT_RECOGNITION_MODELS`` (sepaelv6/KPRPE).
        """
        return try_compute_embeddings_batch(
            self._compute_embeddings_batch,
            bgr_aligned_batch,
            aligned_keypoints_batch=aligned_keypoints_batch,
        )

    def _assemble_result(self, face: FaceData, bgr_aligned_face, embeddings, fiqa_score):
        """Assembles the result dictionary for a face."""
        return build_face_result(
            aligned_face=cv2.cvtColor(bgr_aligned_face, cv2.COLOR_BGR2RGB),
            bbox=face.bbox,
            keypoints=face.kps,
            det_score=face.det_score,
            embeddings=embeddings,
            fiqa_score=fiqa_score,
            models=self.models,
            extended=self.extended,
            concat_embeddings=self.concat_embeddings,
            gender=face.gender,
            age=face.age,
            pose=face.pose,
        )

    def _assemble_result_from_align_only(
        self, align_item, embeddings, fiqa_score
    ):
        """Builds a ``process_image``-compatible result dict from an
        ``align_only`` output plus embeddings extracted in batch.

        Output keys and types match ``_assemble_result`` exactly, so
        callers can treat the two interchangeably.

        Note: ``align_only`` already returns ``aligned_face`` in RGB
        and ``gender`` as ``"M"``/``"F"`` — no conversion here.
        """
        return build_face_result_from_align_result(
            align_item=align_item,
            embeddings=embeddings,
            fiqa_score=fiqa_score,
            models=self.models,
            extended=self.extended,
            concat_embeddings=self.concat_embeddings,
        )

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
        1. Per-image: ``align_only`` (detect + warp), accumulate aligned
           crops into a buffer.
        2. Per-chunk of ``batch_size``: ``_compute_embeddings_batch`` in
           one ONNX call.
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
                may warn while reducing the batch size (see ``_try_compute_embeddings_batch``).

        Returns:
            list: Results parallel to ``imgpaths``. With ``single_face=True``,
            each item is a ``dict`` compatible with ``process_image`` or
            ``None`` when no face is detected. With ``single_face=False``,
            each item is a list of per-image face dictionaries.

        Notes:
            Embeddings produced here are equivalent to those from
            ``process_image`` (same ONNX ops, just batched). On GPU
            the parallel reductions are not bit-exact deterministic,
            so embeddings may differ from the per-image path by tiny
            amounts — cosine similarity stays essentially the same.
        """
        imgpaths = list(imgpaths)
        if not imgpaths:
            return []

        if single_face:
            aligned_items = [
                self.align_only(
                    p,
                    single_face=True,
                    select_single_face_by=select_single_face_by,
                )
                for p in imgpaths
            ]
            results: list = [None] * len(imgpaths)

            valid_indices = [
                i for i, item in enumerate(aligned_items) if item is not None
            ]
            needs_kps = bool(
                set(self.models) & self.KEYPOINT_RECOGNITION_MODELS
            )
            for chunk_start in range(0, len(valid_indices), batch_size):
                chunk_idx = valid_indices[chunk_start : chunk_start + batch_size]
                # `align_only` returns RGB (consistent with `process_image`)
                # but the recognition ONNX session was trained on BGR — flip
                # color order at the boundary instead of inside the alignment.
                crops = np.stack(
                    [
                        cv2.cvtColor(
                            aligned_items[i]["aligned_face"],
                            cv2.COLOR_RGB2BGR,
                        )
                        for i in chunk_idx
                    ],
                    axis=0,
                )
                # Keypoints alinhados só são empilhados quando algum modelo
                # carregado é KEYPOINT_RECOGNITION_MODELS (ex: sepaelv6).
                # Caso contrário, evita custo de stack desnecessário.
                kps_batch = None
                if needs_kps:
                    kps_batch = np.stack(
                        [aligned_items[i]["aligned_keypoints"] for i in chunk_idx],
                        axis=0,
                    )
                embeddings, fiqa_scores = self._try_compute_embeddings_batch(
                    crops, aligned_keypoints_batch=kps_batch,
                )

                for k, idx in enumerate(chunk_idx):
                    if self.concat_embeddings:
                        emb = embeddings[k]
                    else:
                        emb = [per_model[k] for per_model in embeddings]
                    fiqa = (
                        float(fiqa_scores[k])
                        if fiqa_scores is not None
                        else None
                    )
                    results[idx] = self._assemble_result_from_align_only(
                        aligned_items[idx], emb, fiqa
                    )
            return results

        # multi-face: flatten all detected faces, batch over the flat
        # list, scatter the dicts back into per-image bins.
        aligned_per_image = [
            self.align_only(
                p,
                single_face=False,
                select_single_face_by=select_single_face_by,
            )
            for p in imgpaths
        ]
        results_multi: list = [[] for _ in imgpaths]
        flat: list = []
        for img_idx, faces in enumerate(aligned_per_image):
            for face_item in faces:
                flat.append((img_idx, face_item))
        if not flat:
            return results_multi

        needs_kps = bool(
            set(self.models) & self.KEYPOINT_RECOGNITION_MODELS
        )
        for chunk_start in range(0, len(flat), batch_size):
            chunk = flat[chunk_start : chunk_start + batch_size]
            # RGB → BGR for ONNX (see corresponding comment in single_face path).
            crops = np.stack(
                [
                    cv2.cvtColor(face_item["aligned_face"], cv2.COLOR_RGB2BGR)
                    for _, face_item in chunk
                ],
                axis=0,
            )
            kps_batch = None
            if needs_kps:
                kps_batch = np.stack(
                    [face_item["aligned_keypoints"] for _, face_item in chunk],
                    axis=0,
                )
            embeddings, fiqa_scores = self._try_compute_embeddings_batch(
                crops, aligned_keypoints_batch=kps_batch,
            )

            for k, (img_idx, face_item) in enumerate(chunk):
                if self.concat_embeddings:
                    emb = embeddings[k]
                else:
                    emb = [per_model[k] for per_model in embeddings]
                fiqa = (
                    float(fiqa_scores[k]) if fiqa_scores is not None else None
                )
                results_multi[img_idx].append(
                    self._assemble_result_from_align_only(face_item, emb, fiqa)
                )

        return results_multi

    def _load_image(self, imgpath):
        """Load image from file path or return the array if already loaded."""
        return cv2.imread(imgpath) if isinstance(imgpath, str) else imgpath.copy()

    def process_image_multiple_faces(
        self, imgpath: str, draw_keypoints=False  # Path to image to be processed
    ):
        """
        Process an image with one or multiple faces.
        THIS METHOD IS DEPRECATED AND WILL BE REMOVED IN A FUTURE RELEASE. Use process_image instead.
        """
        warnings.warn(
            "process_image_multiple_faces: This method is deprecated and will be removed in a future release.\n"
            "Use the 'process_image' method instead.",
            DeprecationWarning,
        )
        bgr_img = self._load_image(imgpath)
        return self.process_image(
            bgr_img, draw_keypoints=draw_keypoints, single_face=False
        )

    def build_mosaic(
        self,
        img_path_list: list[str | np.ndarray],
        mosaic_shape: tuple[int, int],
        border: float = 0.03,
        save_to: str | None = None,
        draw_keypoints: bool = False,
    ) -> np.ndarray:
        """
        Build a rectangular mosaic of the aligned faces.
        Based on the imutils build_montages function.

        Args:
            img_path_list: list of paths to image files or list of bgr_images
            mosaic_shape: tuple of integers, (n_cols, n_rows)
            border: float, percent of image to use as white border
            save_to: optional path used to save the mosaic image.
            draw_keypoints: if True, draw keypoints on each aligned face.

        Returns:
            np.ndarray: OpenCV BGR image with mosaic.
        """
        return build_aligned_face_mosaic(
            self,
            img_path_list,
            mosaic_shape,
            border=border,
            save_to=save_to,
            draw_keypoints=draw_keypoints,
        )

    def compare(self, img1path: str, img2path: str) -> float:
        """
        Compares the similarity between two face images based on their embeddings.

        Args:
            img1path: Path to the first image file.
            img2path: Path to the second image file.

        Returns:
            float: Similarity score between the two faces based on their embeddings.
            The score ranges from -1.0 to 1.0, where 1.0 represents a perfect match and -1.0 represents a complete mismatch.

        Raises:
            ValueError: If ``concat_embeddings`` is False, because this method
                requires a single concatenated embedding for each image.
        """
        if not self.concat_embeddings:
            raise ValueError(
                "compare() is not compatible with concat_embeddings=False. "
                "Instantiate ForensicFace with concat_embeddings=True, or "
                "compare the model-specific embedding_<model_name> arrays manually."
            )

        img1data = self.process_image(img1path, single_face=True)
        assert len(img1data) > 0, f"No face detected in {img1path}"
        img2data = self.process_image(img2path, single_face=True)
        assert len(img2data) > 0, f"No face detected in {img2path}"

        return np.dot(img1data["embedding"], img2data["embedding"]) / (
            np.linalg.norm(img1data["embedding"]) * np.linalg.norm(img2data["embedding"])
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
        if weights is None:
            weights = np.ones(embeddings.shape[0], dtype="int")
        assert embeddings.shape[0] == weights.shape[0]
        assert method in ["mean", "median"]
        if method == "mean":
            return np.average(embeddings, axis=0, weights=weights)
        else:
            weighted_embeddings = np.array([w * e for w, e in zip(weights, embeddings)])
            return np.median(weighted_embeddings, axis=0)

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
        if quality_weight:
            assert (
                self.extended == True
            ), "You must initialize ForensicFace with extended = True"

        if self.concat_embeddings:
            embeddings = []
        else:
            embeddings = {model_name: [] for model_name in self.models}
        weights = []
        for imgpath in list_of_image_paths:
            d = self.process_image(imgpath, single_face=True)
            if len(d) > 0:
                if self.concat_embeddings:
                    embeddings.append(d["embedding"])
                else:
                    for model_name in self.models:
                        embeddings[model_name].append(d[f"embedding_{model_name}"])
                weights.append(d["fiqa_score"] if quality_weight == True else 1.0)
        if len(weights) > 0:
            weights_array = np.array(weights)
            if not self.concat_embeddings:
                return {
                    f"embedding_{model_name}": self.aggregate_embeddings(
                        np.array(model_embeddings),
                        method=method,
                        weights=weights_array,
                    )
                    for model_name, model_embeddings in embeddings.items()
                }
            return self.aggregate_embeddings(
                np.array(embeddings), method=method, weights=weights_array
            )
        else:
            return []

    def _get_extended_bbox(self, bbox, frame_shape, margin_factor):
        """
        Computes and returns the bounding box with extended margins.

        Parameters:
            bbox (ndarray): The bounding box coordinates (startX, startY, endX, endY).
            frame_shape (tuple): The shape of the video frame (height, width, channels).
            margin_factor (float): The factor to be applied for computing the margin.

        Returns:
            A list with the coordinates of the extended bounding box (startX_out, startY_out, endX_out, endY_out).
        """
        return extend_bbox(bbox, frame_shape, margin_factor)

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
        assert rgb_aligned_face.shape == (*self.IMG_SIZE, 3)
        bgr_aligned_face = rgb_aligned_face[..., ::-1].copy()
        embeddings, fiqa_score = self._compute_embeddings(
            bgr_aligned_face, aligned_keypoints=keypoints
        )

        if self.concat_embeddings:
            ret = FaceResult({"embedding": embeddings})
        else:
            ret = FaceResult()
            for model_name, embedding in zip(self.models, embeddings):
                ret["embedding_" + model_name] = embedding
        if self.extended:
            ret["fiqa_score"] = fiqa_score
        return ret
