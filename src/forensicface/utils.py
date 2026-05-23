__all__ = [
    'aggregate_embeddings',
    'cosine_score',
    'cosine_similarity',
    'compute_ss_ds',
    'freeze_env',
    'transform_keypoints',
    'annotate_img_with_kps',
]
import numpy as np

def cosine_similarity(X, Z):
    # Normalize the embeddings
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)

    # Compute the dot product between the normalized embeddings
    dot_product = np.dot(X, Z.T)

    # Return the cosine similarity between the embeddings
    return dot_product

def cosine_score(x: np.ndarray, z: np.ndarray) -> float:
    """Compute cosine similarity between two 1D embeddings."""
    return np.dot(x, z) / (np.linalg.norm(x) * np.linalg.norm(z))

def aggregate_embeddings(
    embeddings: np.ndarray,
    weights: np.ndarray | None = None,
    method: str = "mean",
) -> np.ndarray:
    """
    Aggregates multiple embeddings into a single embedding.

    Args:
        embeddings: A 2D array of shape (num_embeddings, embedding_dim)
            containing the embeddings to be aggregated.
        weights: A 1D array of shape (num_embeddings,) containing the weights
            to assign to each embedding. If not provided, all embeddings are
            equally weighted.
        method: Aggregation method. Possible values are ``"mean"`` and
            ``"median"``.

    Returns:
        np.ndarray: A 1D array of shape (embedding_dim,) containing the
        aggregated embedding.
    """
    if weights is None:
        weights = np.ones(embeddings.shape[0], dtype="int")
    assert embeddings.shape[0] == weights.shape[0]
    assert method in ["mean", "median"]
    if method == "mean":
        return np.average(embeddings, axis=0, weights=weights)
    weighted_embeddings = np.array([w * e for w, e in zip(weights, embeddings)])
    return np.median(weighted_embeddings, axis=0)

def compute_ss_ds(
    X: np.ndarray,
    x_id: np.ndarray,
    x_names: np.ndarray | None = None,
    Z: np.ndarray | None = None,
    z_id: np.ndarray | None = None,
    z_names: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple] | None]:
    """
    Compute cosine similarities between the cartesian product of two arrays X and Z and
    return same-source (ss) and different-source (ds) scores.
    If only the array X and x_id are provided, compute the cosine similarities between all pairwise
    combination in X. Also return the names of the files associated with each score, is x_names and z_names are provided.

    Args:
        X: 2D numpy array with one embedding per row.
        x_id: 1D numpy array with identity labels for ``X``.
        x_names: Optional 1D numpy array with names of files associated with
            the embeddings in ``X``.
        Z: Optional 2D numpy array with one embedding per row.
        z_id: Optional 1D numpy array with identity labels for ``Z``.
        z_names: Optional 1D numpy array with names of files associated with
            the embeddings in ``Z``.

    Returns:
        tuple[np.ndarray, np.ndarray, list[tuple] | None]: Scores, same-source
        and different-source labels, and optional file-name pairs.
    """
    assert X.ndim == 2
    assert X.shape[0] == len(x_id)
    ss_names = None
    ds_names = None
    if Z is None:  # compute scores of X vs X
        similarities = cosine_similarity(X, X)
        ss_mask = x_id[:, np.newaxis] == x_id
        upper_triangle_mask = np.triu(np.ones_like(similarities), k=1).astype(bool)
        ss = similarities[(ss_mask & upper_triangle_mask)]
        ds = similarities[(~ss_mask & upper_triangle_mask)]
        if x_names is not None:  # compute names of X vs X
            assert X.shape[0] == len(x_id) == len(x_names)
            ss_names = [
                (x_names[i], x_names[j])
                for i, j in np.argwhere(upper_triangle_mask)
                if x_id[i] == x_id[j]
            ]
            ds_names = [
                (x_names[i], x_names[j])
                for i, j in np.argwhere(upper_triangle_mask)
                if x_id[i] != x_id[j]
            ]
    if Z is not None:  # compute scores of X vs Z
        assert Z.ndim == 2
        assert Z.shape[0] == len(z_id)
        similarities = cosine_similarity(X, Z)
        ss_mask = x_id[:, np.newaxis] == z_id
        ss = similarities[ss_mask]
        ds = similarities[~ss_mask]
        if z_names is not None:  # compute names of X vs Z
            assert Z.shape[0] == len(z_id) == len(z_names)
            ss_names = [(x_names[i], z_names[j]) for i, j in np.argwhere(ss_mask)]
            ds_names = [(x_names[i], z_names[j]) for i, j in np.argwhere(~ss_mask)]

    scores = np.concatenate([ss, ds])
    y = np.concatenate([np.ones(len(ss)), np.zeros(len(ds))])
    names = (
        ss_names + ds_names if ss_names is not None and ds_names is not None else None
    )
    return scores, y, names

def freeze_env():
    import sys
    from importlib.metadata import distributions

    env = {}
    env.update({"Python version": f"{sys.version}"})

    installed_packages = [
        (d.metadata.get("Name", d.name), d.version) for d in distributions()
    ]
    installed_packages.sort(
        key=lambda x: x[0].lower()
    )  # Sort alphabetically, case-insensitive

    for package, version in installed_packages:
        env.update({f"{package}": f"{version}"})
    return env

def transform_keypoints(keypoints: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Transforms keypoints from the original image space to the aligned image space.

    Args:
        keypoints: A 2D array of shape (5, 2) representing the original keypoints.
        M: The 2x3 affine transformation matrix.

    Returns:
        np.ndarray: A 2D array of shape (5, 2) representing the transformed keypoints.
    """
    # Add a third dimension of ones to keypoints to allow affine transformation
    keypoints_homo = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
    transformed_keypoints = (M @ keypoints_homo.T).T  # Apply affine transformation
    return transformed_keypoints

def annotate_img_with_kps(
    bgr_img: np.ndarray, kps: np.ndarray, color: str = "red", radius: int = 2
) -> np.ndarray:
    """
    Annotate an image with keypoints.

    Parameters:
    bgr_img (numpy.ndarray): The input image in BGR format.
    kps (numpy.ndarray): A numpy array of shape (5, 2) containing the keypoints.
    color (str, optional): The color of the keypoints. Default is 'red'.
                        Options are 'red', 'blue', 'green', 'white', 'black'.
    radius (int, optional): The radius of the keypoints. Default is 2.

    Returns:
    numpy.ndarray: The image with keypoints annotated.
    """
    import cv2

    colors = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }

    assert color in colors.keys()
    assert kps.shape == (5, 2)

    bgr_img_with_kps = bgr_img.copy()

    for x, y in kps:
        cv2.circle(
            bgr_img_with_kps,
            (int(x), int(y)),
            radius=radius,
            color=colors[color],
            thickness=-1,
        )

    return bgr_img_with_kps
