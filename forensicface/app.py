# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_forensicface.ipynb.

# %% auto 0
__all__ = ['ForensicFace']

# %% ../nbs/00_forensicface.ipynb 2
from nbdev.showdoc import *
from fastcore.utils import *
import onnxruntime
import cv2
import numpy as np
import os.path as osp
from insightface.app import FaceAnalysis
from insightface.utils import face_align


# %% ../nbs/00_forensicface.ipynb 3
class ForensicFace:
    "A (forensic) face comparison tool"

    def __init__(
        self, model: str = "sepaelv2", det_size: int = 320, use_gpu: bool = True
    ):

        self.det_size = (det_size, det_size)

        # model_base_path = osp.join(osp.expanduser("~/.insightface/models"), model)
        # adaface_model_folder = osp.join(model_base_path, "adaface")
        # det_path = osp.join(model_base_path, "det_10g.onnx")
        # rec_path = osp.join(adaface_model_folder, "adaface_ir101web12m.onnx")

        # if not osp.exists(det_path):
        #    pass

        # if not osp.exists(det_path):
        #    pass

        self.detectmodel = FaceAnalysis(
            name=model,
            # allowed_modules=["detection","landmark_3d_68","genderage"],
            providers=["CUDAExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"],
        )
        self.detectmodel.prepare(ctx_id=0 if use_gpu else -1, det_size=self.det_size)
        self.ort_ada = onnxruntime.InferenceSession(
            osp.join(
                osp.expanduser("~/.insightface/models"),
                model,
                "adaface",
                "adaface_ir101web12m.onnx",
            ),
            providers=["CUDAExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"],
        )

        self.ort_mag = onnxruntime.InferenceSession(
            osp.join(
                osp.expanduser("~/.insightface/models"),
                model,
                "magface",
                "magface_iresnet100.onnx",
            ),
            providers=["CUDAExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"],
        )

    def _to_input_ada(self, aligned_bgr_img):
        _aligned_bgr_img = aligned_bgr_img.astype(np.float32)
        _aligned_bgr_img = ((_aligned_bgr_img / 255.0) - 0.5) / 0.5
        return _aligned_bgr_img.transpose(2, 0, 1).reshape(1, 3, 112, 112)

    def _to_input_mag(self, aligned_bgr_img):
        _aligned_bgr_img = aligned_bgr_img.astype(np.float32)
        _aligned_bgr_img = _aligned_bgr_img / 255.0
        return _aligned_bgr_img.transpose(2, 0, 1).reshape(1, 3, 112, 112)

    def get_most_central_face(self, img, faces):
        """
        faces is a insightface object with keypoints and bounding_box

        return: keypoints of the most central face
        """
        assert faces is not None
        img_center = np.array([img.shape[0] // 2, img.shape[1] // 2])
        dist = []

        # Compute centers of faces and distances from certer of image
        for idx, face in enumerate(faces):
            box = face.bbox.astype("int").flatten()
            face_center = np.array([(box[0] + box[2]) // 2, (box[1] + box[3]) // 2])
            dist.append(np.linalg.norm(img_center - face_center))

        # Get index of the face closest to the center of image
        idx = dist.index(min(dist))
        return idx, faces[idx].kps

    def process_image_single_face(self, imgpath: str):  # Path to image to be processed
        """
        Process image and returns dict with:

        - keypoints: 5 facial points (left eye, right eye, nose tip, left mouth corner and right mouth corner)

        - ipd: interpupillary distance

        - normalized_embedding

        - embedding_norm

        - aligned_face: face after alignment using the keypoints as references for affine transform
        """
        bgr_img = cv2.imread(imgpath)
        faces = self.detectmodel.get(bgr_img)
        if len(faces) == 0:
            return {}
        idx, kps = self.get_most_central_face(bgr_img, faces)
        gender = "M" if faces[idx].gender == 1 else "F"
        age = faces[idx].age
        pitch, yaw, roll = faces[idx].pose
        bgr_aligned_face = face_align.norm_crop(bgr_img, kps)
        ipd = np.linalg.norm(kps[0] - kps[1])
        ada_inputs = {
            self.ort_ada.get_inputs()[0].name: self._to_input_ada(bgr_aligned_face)
        }
        mag_inputs = {
            self.ort_mag.get_inputs()[0].name: self._to_input_mag(bgr_aligned_face)
        }
        normalized_embedding, norm = self.ort_ada.run(None, ada_inputs)
        mag_embedding = self.ort_mag.run(None, ada_inputs)[0][0]
        mag_norm = np.linalg.norm(mag_embedding)

        return {
            "keypoints": kps,
            "ipd": ipd,
            "gender": gender,
            "age": age,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "embedding": normalized_embedding.flatten() * norm.flatten()[0],
            "norm": norm.flatten()[0],
            "magface_embedding": mag_embedding,
            "magface_norm": mag_norm,
            "aligned_face": cv2.cvtColor(bgr_aligned_face, cv2.COLOR_BGR2RGB),
        }

    def process_image(self, imgpath):
        return self.process_image_single_face(imgpath)
        
    def process_image_multiple_faces(
        self, imgpath: str, # Path to image to be processed

    ):  
        """
        Process image and returns list of dicts with:

        - keypoints: 5 facial points (left eye, right eye, nose tip, left mouth corner and right mouth corner)

        - ipd: interpupillary distance

        - normalized_embedding

        - embedding_norm

        - aligned_face: face after alignment using the keypoints as references for affine transform
        """
        bgr_img = cv2.imread(imgpath)
        faces = self.detectmodel.get(bgr_img)
        if len(faces) == 0:
            return []
        ret = []
        for face in faces:

            kps = face.kps
            gender = "M" if face.gender == 1 else "F"
            age = face.age
            pitch, yaw, roll = face.pose
            bgr_aligned_face = face_align.norm_crop(bgr_img, kps)
            ipd = np.linalg.norm(kps[0] - kps[1])
            ada_inputs = {
                self.ort_ada.get_inputs()[0].name: self._to_input_ada(bgr_aligned_face)
            }
            #mag_inputs = {
            #    self.ort_mag.get_inputs()[0].name: self._to_input_mag(bgr_aligned_face)
            #}
            normalized_embedding, norm = self.ort_ada.run(None, ada_inputs)
            #mag_embedding = self.ort_mag.run(None, ada_inputs)[0][0]
            #mag_norm = np.linalg.norm(mag_embedding)

            ret.append(
                {
                    "keypoints": kps,
                    "ipd": ipd,
                    "gender": gender,
                    "age": age,
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll,
                    "embedding": normalized_embedding.flatten() * norm.flatten()[0],
                    "norm": norm.flatten()[0],
                   #"magface_embedding": mag_embedding,
                    #"magface_norm": mag_norm,
                    #"aligned_face": cv2.cvtColor(bgr_aligned_face, cv2.COLOR_BGR2RGB),
                }
            )
        return ret


# %% ../nbs/00_forensicface.ipynb 8
@patch
def compare(self: ForensicFace, img1path: str, img2path: str):
    img1data = self.process_image(img1path)
    assert len(img1data) > 0
    img2data = self.process_image(img2path)
    assert len(img2data) > 0
    return np.dot(img1data["embedding"], img2data["embedding"]) / (
        img1data["norm"] * img2data["norm"]
    )


# %% ../nbs/00_forensicface.ipynb 11
@patch
def aggregate_embeddings(self: ForensicFace, embeddings, weights=None):
    if weights is None:
        weights = np.ones(embeddings.shape[0], dtype="int")
    assert embeddings.shape[0] == weights.shape[0]
    return np.average(embeddings, axis=0, weights=weights)


# %% ../nbs/00_forensicface.ipynb 12
@patch
def aggregate_from_images(self: ForensicFace, list_of_image_paths):
    embeddings = []
    weights = []
    for imgpath in list_of_image_paths:
        d = self.process_image(imgpath)
        embeddings.append(d["embedding"])
    return self.aggregate_embeddings(np.array(embeddings))

