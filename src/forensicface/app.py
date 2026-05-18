__all__ = ['custom_formatwarning', 'ForensicFace']
import os
import onnxruntime
import cv2
import numpy as np
import os.path as osp
from glob import glob
from imutils import build_montages
from tqdm import tqdm
import warnings
from .backends import FaceData, FaceBackend, create_backend
from .utils import freeze_env, transform_keypoints, annotate_img_with_kps
from .ort_runtime_setup import configure_onnxruntime_acceleration
from .runtime_summary import print_initialization_summary


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
                osp.join(
                    self.models_root,
                    self.models[0],
                    "cr_fiqa",
                    "cr_fiqa_l.onnx",
                ),
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
        """Loads a single ONNX model."""
        model_path = glob(
            osp.join(
                models_root, model_name, "*", "*face*.onnx"
            )
        )
        if len(model_path) == 0:
            raise Exception(f"No face embedding model found in {osp.join(models_root, model_name)}")
        if len(model_path) > 1:
            raise Exception(f"Multiple face embedding models found in {osp.join(models_root, model_name)}: {model_path}\nPlease ensure there is only one ONNX file for face embedding in the model directory.")
        return onnxruntime.InferenceSession(
            model_path[0],
            providers=providers,
        )

    def _to_input_ada(self, aligned_bgr_img):
        """
        Preprocesses the input face for the face recognition model.

        Args:
            face: Face image as a numpy array in BGR order.

        Returns:
            Preprocessed face image as a numpy array.
        """
        _aligned_bgr_img = aligned_bgr_img.astype(np.float32)
        _aligned_bgr_img = ((_aligned_bgr_img / 255.0) - 0.5) / 0.5
        return _aligned_bgr_img.transpose(2, 0, 1).reshape(1, 3, *self.IMG_SIZE)

    def _get_best_face(self, img, faces, criterion="size"):
        """Get the best face based on a criterion: 'centrality' or 'size'."""
        assert criterion in ["centrality", "size"]
        assert faces is not None and len(faces) > 0

        if criterion == "centrality":
            img_center = np.array([img.shape[0] // 2, img.shape[1] // 2])
            scores = [
                np.linalg.norm(
                    img_center
                    - np.array([(box[0] + box[2]) // 2, (box[1] + box[3]) // 2])
                )
                for box in [face.bbox.astype("int").flatten() for face in faces]
            ]
        elif criterion == "size":
            scores = [
                abs((box[2] - box[0]) * (box[3] - box[1]))
                for box in [face.bbox.astype("int").flatten() for face in faces]
            ]

        if criterion == "centrality":
            best_idx = scores.index(min(scores))
        else:
            best_idx = scores.index(max(scores))

        return faces[best_idx]

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
        img_to_input = self._to_input_ada(bgr_aligned_face)
        embeddings = []
        for model_name, rec_ort in zip(self.models, self.rec_inference_sessions):
            if model_name in self.KEYPOINT_RECOGNITION_MODELS:
                model_inputs = self._build_keypoint_model_inputs(
                    model_name=model_name,
                    rec_ort=rec_ort,
                    img_to_input=img_to_input,
                    aligned_keypoints=aligned_keypoints,
                )
            else:
                model_inputs = {rec_ort.get_inputs()[0].name: img_to_input}
            model_output = rec_ort.run(None, model_inputs)
            if (
                len(model_output) == 2
            ):  # model output in the form of normed_embedding, norm
                embedding = model_output[0].flatten() * model_output[1].flatten()[0]
            else:  # model output in the form of embedding
                embedding = model_output[0].flatten()
            embeddings.append(embedding)

        fiqa_score = None
        if self.extended:
            _, fiqa_score = self.ort_fiqa.run(
                None, {self.ort_fiqa.get_inputs()[0].name: img_to_input}
            )

        return (
            np.concatenate(embeddings) if self.concat_embeddings else embeddings,
            fiqa_score[0][0] if fiqa_score is not None else None,
        )

    def _build_keypoint_model_inputs(
        self, model_name, rec_ort, img_to_input, aligned_keypoints
    ):
        input_names = [input_info.name for input_info in rec_ort.get_inputs()]
        required_input_names = {
            self.SEPAELV6_IMAGE_INPUT,
            self.SEPAELV6_KEYPOINTS_INPUT,
        }
        missing_input_names = required_input_names.difference(input_names)
        if missing_input_names:
            raise ValueError(
                f"Model '{model_name}' requires ONNX inputs "
                f"{sorted(required_input_names)}, but the loaded session has "
                f"{input_names}. Missing: {sorted(missing_input_names)}."
            )

        return {
            self.SEPAELV6_IMAGE_INPUT: img_to_input,
            self.SEPAELV6_KEYPOINTS_INPUT: self._to_keypoints_input(
                aligned_keypoints, model_name=model_name
            ),
        }

    def _to_keypoints_input(self, aligned_keypoints, model_name):
        if aligned_keypoints is None:
            raise ValueError(
                f"Model '{model_name}' requires aligned 5-point keypoints in "
                "the 112x112 face image coordinate system."
            )

        aligned_keypoints = np.asarray(aligned_keypoints, dtype=np.float32)
        if aligned_keypoints.shape != (5, 2):
            raise ValueError(
                f"Model '{model_name}' requires keypoints with shape (5, 2); "
                f"received {aligned_keypoints.shape}."
            )

        normalized_keypoints = aligned_keypoints.copy()
        normalized_keypoints[:, 0] /= self.IMG_SIZE[1]
        normalized_keypoints[:, 1] /= self.IMG_SIZE[0]
        return normalized_keypoints.reshape(1, 5, 2)

    def _assemble_result(self, face: FaceData, bgr_aligned_face, embeddings, fiqa_score):
        """Assembles the result dictionary for a face."""
        ret = {
            "ipd": np.linalg.norm(face.kps[0] - face.kps[1]),
        }

        if self.extended:
            gender = None
            if face.gender is not None:
                gender = "M" if face.gender == 1 else "F"

            yaw, pitch, roll = None, None, None
            if face.pose is not None:
                yaw = face.pose[1]
                pitch = face.pose[0]
                roll = face.pose[2]

            ret.update(
                {
                    "fiqa_score": fiqa_score,
                    "gender": gender,
                    "age": face.age,
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll,
                }
            )

        ret.update(
            {
                "det_score": face.det_score,
                "keypoints": face.kps,
                "bbox": face.bbox.astype("int"),
            }
        )
        if self.concat_embeddings:
            ret["embedding"] = embeddings
        else:
            for model_name, embedding in zip(self.models, embeddings):
                ret[f"embedding_{model_name}"] = embedding

        ret["aligned_face"] = cv2.cvtColor(bgr_aligned_face, cv2.COLOR_BGR2RGB)
        return ret

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
        img_path_list,
        mosaic_shape,
        border=0.03,
        save_to=None,
        draw_keypoints=False,
    ):
        """
        Build a rectangular mosaic of the aligned faces.
        Based on the imutils build_montages function.

        Parameters:
            img_path_list: list of paths to image files or list of bgr_images
            mosaic_shape: tuple of integers, (n_cols, n_rows)
            border: float, percent of image to use as white border

        Returns:
            cv2 BGR image with mosaic
        """
        assert mosaic_shape is not None
        top = int(border * self.IMG_SIZE[0])  # shape[0] = rows
        bottom = top
        left = int(border * self.IMG_SIZE[1])  # shape[1] = cols
        right = left

        imgs = []
        list_of_arrays = False
        for img in img_path_list:
            if type(img) != str:  # image array passed as argument
                list_of_arrays = True
            ret = self.process_image(
                img, draw_keypoints=draw_keypoints, single_face=True
            )
            if len(ret) > 0:
                img = cv2.cvtColor(ret["aligned_face"], cv2.COLOR_RGB2BGR)
                img = cv2.copyMakeBorder(
                    img,
                    top=top,
                    bottom=bottom,
                    left=left,
                    right=right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )
                imgs.append(img)
        mosaic = build_montages(
            imgs,
            image_shape=(
                int(self.IMG_SIZE[0] * (1 + 2 * border)),
                int(self.IMG_SIZE[1] * (1 + 2 * border)),
            ),
            montage_shape=mosaic_shape,
        )[0]
        if list_of_arrays:
            warnings.warn(
                "A list of arrays was passed as argument. Make sure image arrays are in BGR format.",
                Warning,
            )
        if save_to is not None:
            cv2.imwrite(save_to, mosaic)
        return mosaic

    def compare(self, img1path: str, img2path: str):
        """
        Compares the similarity between two face images based on their embeddings.

        Parameters:
            - img1path (str): Path to the first image file
            - img2path (str): Path to the second image file

        Returns:
            A float representing the similarity score between the two faces based on their embeddings.
            The score ranges from -1.0 to 1.0, where 1.0 represents a perfect match and -1.0 represents a complete mismatch.
        """
        img1data = self.process_image(img1path, single_face=True)
        assert len(img1data) > 0, f"No face detected in {img1path}"
        img2data = self.process_image(img2path, single_face=True)
        assert len(img2data) > 0, f"No face detected in {img2path}"
        assert self.concat_embeddings == True

        return np.dot(img1data["embedding"], img2data["embedding"]) / (
            np.linalg.norm(img1data["embedding"]) * np.linalg.norm(img2data["embedding"])
        )

    def aggregate_embeddings(self, embeddings, weights=None, method="mean"):
        """
        Aggregates multiple embeddings into a single embedding.

        Args:
            embeddings (numpy.ndarray): A 2D array of shape (num_embeddings, embedding_dim) containing the embeddings to be
                aggregated.
            weights (numpy.ndarray, optional): A 1D array of shape (num_embeddings,) containing the weights to be assigned
                to each embedding. If not provided, all embeddings are equally weighted.

            method (str, optional): choice of agregating based on the mean or median of the embeddings. Possible values are
                'mean' and 'median'.

        Returns:
            numpy.ndarray: A 1D array of shape (embedding_dim,) containing the aggregated embedding.
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
        self, list_of_image_paths, method="mean", quality_weight=False
    ):
        """
        Given a list of image paths, this method returns the average embedding of all faces found in the images.

        Args:
            list_of_image_paths (List[str]): List of paths to images.
            method (str, optional): choice of agregating based on the mean or median of the embeddings. Possible values are
                'mean' and 'median'.
            quality_weight (boolean, optional): If True, use the FIQA(L) score as a weight for aggregation.

        Returns:
            Union[np.ndarray, List]: If one or more faces are found, returns a 1D numpy array of shape (512,) representing the
            average embedding. Otherwise, returns an empty list.
        """
        if quality_weight:
            assert (
                self.extended == True
            ), "You must initialize ForensicFace with extended = True"
        assert self.concat_embeddings == True
        embeddings = []
        weights = []
        for imgpath in list_of_image_paths:
            d = self.process_image(imgpath, single_face=True)
            if len(d) > 0:
                embeddings.append(d["embedding"])
                weights.append(d["fiqa_score"] if quality_weight == True else 1.0)
        if len(embeddings) > 0:
            return self.aggregate_embeddings(
                np.array(embeddings), method=method, weights=np.array(weights)
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
        # add a margin on the bounding box
        (startX, startY, endX, endY) = bbox.astype("int")
        (h, w) = frame_shape[:2]
        out_width = (endX - startX) * margin_factor
        out_height = (endY - startY) * margin_factor

        startX_out = int((startX + endX) / 2 - out_width / 2)
        endX_out = int((startX + endX) / 2 + out_width / 2)
        startY_out = int((startY + endY) / 2 - out_height / 2)
        endY_out = int((startY + endY) / 2 + out_height / 2)

        # tests if the output bbox coordinates are out of frame limits
        if startX_out < 0:
            startX_out = 0
        if endX_out > int(w):
            endX_out = int(w)
        if startY_out < 0:
            startY_out = 0
        if endY_out > int(h):
            endY_out = int(h)
        return [startX_out, startY_out, endX_out, endY_out]

    def extract_faces(
        self,
        video_path: str,  # path to video file
        dest_folder: str = None,  # folder used to save extracted faces. If not provided, a new folder with the video name is created
        every_n_frames: int = 1,  # skip some frames
        margin: float = 2.0,  # margin to add to each face, w.r.t. detected bounding box
        start_from: float = 0.0,  # seconds after video start to begin processing
        export_metadata: bool = False,  # if True, export facial keypoints, bounding box, ipd, fiqa_score, pitch, yaw, roll, and embedding
    ):
        """
        Extracts faces from a video and saves them as individual images.

        Parameters:
            video_path (str): The path to the input video file.
            dest_folder (str, optional): The path to the output folder. If not provided, a new folder with the same name as the input video file is created.
            every_n_frames (int, optional): Extract faces from every n-th frame. Default is 1 (extract faces from all frames).
            margin (float, optional): The factor by which the detected face bounding box should be extended. Default is 2.0.
            start_from (float, optional): The time point (in seconds) after which the video frames should be processed. Default is 0.0.

        Returns:
            The number of extracted faces.
        """
        import pandas as pd

        if dest_folder is None:
            dest_folder = os.path.splitext(video_path)[0]

        os.makedirs(dest_folder, exist_ok=True)

        # initialize video stream from file
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        start_frame = int(fps * start_from)
        total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT)) // every_n_frames

        # seek to starting frame
        vs.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        nfaces = 0
        if export_metadata:
            metadata = []
        with tqdm(
            total=total_frames,
            bar_format="Frames processed: {n}/{total} | Time elapsed: {elapsed}",
        ) as pbar:
            while True:

                ret, frame = vs.read()

                if not ret:
                    break

                current_frame = current_frame + 1
                if (current_frame % every_n_frames) != 0:
                    continue

                (h, w) = frame.shape[:2]

                # faces are provided by the configured backend
                rets = self.process_image(frame, single_face=False)
                for i, ret in enumerate(rets):
                    startX, startY, endX, endY = ret["bbox"]
                    faceW = endX - startX
                    faceH = endY - startY
                    outBbox = self._get_extended_bbox(
                        ret["bbox"], frame.shape, margin_factor=margin
                    )
                    # export the face (with added margin)
                    face_crop = frame[outBbox[1] : outBbox[3], outBbox[0] : outBbox[2]]
                    face_img_path = os.path.join(
                        dest_folder, f"frame_{current_frame:07}_face_{i:02}.png"
                    )
                    cv2.imwrite(face_img_path, face_crop)
                    if export_metadata:
                        metadata.append(
                            {
                                **{"frame": current_frame, "face": i},
                                **{
                                    k: v
                                    for k, v in ret.items()
                                    if k not in ["det_score", "aligned_face"]
                                },
                            }
                        )
                    nfaces += 1
                pbar.update(1)
        vs.release()
        if export_metadata:
            pd.DataFrame(metadata).to_json(
                os.path.join(
                    dest_folder,
                    os.path.splitext(os.path.basename(video_path))[0] + ".jsonl",
                ),
                lines=True,
                orient="records",
            )
        return nfaces

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
            ret = {"embedding": embeddings}
        else:
            ret = {}
            for model_name, embedding in zip(self.models, embeddings):
                ret["embedding_" + model_name] = embedding
        if self.extended:
            ret = {**ret, **{"fiqa_score": fiqa_score}}
        return ret
