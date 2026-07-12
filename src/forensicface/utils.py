__all__ = [
    'aggregate_embeddings',
    'cosine_score',
    'cosine_similarity',
    'compute_ss_ds',
    'freeze_env',
    'transform_keypoints',
    'annotate_img_with_kps',
    'DEFAULT_KEYPOINT_COLORS',
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
    if embeddings.shape[0] != weights.shape[0]:
        raise ValueError(
            "weights must have one value per embedding; "
            f"got {weights.shape[0]} weights for {embeddings.shape[0]} embeddings."
        )
    if method not in ["mean", "median"]:
        raise ValueError("method must be either 'mean' or 'median'.")
    if method == "mean":
        return np.average(embeddings, axis=0, weights=weights)
    weighted_embeddings = np.array([w * e for w, e in zip(weights, embeddings)])
    return np.median(weighted_embeddings, axis=0)

def compute_ss_ds(
    X: np.ndarray,
    x_id: np.ndarray,
    Z: np.ndarray | None = None,
    z_id: np.ndarray | None = None,
    return_pair_indices: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Compute cosine similarities between the cartesian product of two arrays X and Z and
    return same-source (ss) and different-source (ds) scores.
    If only the array X and x_id are provided, compute the cosine similarities between all pairwise
    combination in X.

    Args:
        X: 2D numpy array with one embedding per row.
        x_id: 1D numpy array with identity labels for ``X``.
        Z: Optional 2D numpy array with one embedding per row.
        z_id: Optional 1D numpy array with identity labels for ``Z``.
        return_pair_indices: Whether to return ``int32`` pair indices as the
            third return value. If ``False``, pair indices are not computed and
            the third return value is ``None``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray | None]: Scores, same-source
        and different-source labels, and optional ``int32`` pair indices. For
        ``X`` vs ``X``, both index columns refer to rows in ``X``. For ``X`` vs
        ``Z``, the first column refers to ``X`` and the second column refers to
        ``Z``.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array; got ndim={X.ndim}.")
    if X.shape[0] != len(x_id):
        raise ValueError(
            f"x_id length must match X rows; got {len(x_id)} labels for {X.shape[0]} rows."
        )
    if Z is None:  # compute scores of X vs X
        similarities = cosine_similarity(X, X)
        if return_pair_indices:
            pair_i, pair_j = np.triu_indices(X.shape[0], k=1)
            pair_i = pair_i.astype(np.int32, copy=False)
            pair_j = pair_j.astype(np.int32, copy=False)
            ss_pair_mask = x_id[pair_i] == x_id[pair_j]
            ds_pair_mask = ~ss_pair_mask
            ss = similarities[pair_i[ss_pair_mask], pair_j[ss_pair_mask]]
            ds = similarities[pair_i[ds_pair_mask], pair_j[ds_pair_mask]]
            pair_indices = np.empty((len(pair_i), 2), dtype=np.int32)
            n_ss = np.count_nonzero(ss_pair_mask)
            pair_indices[:n_ss, 0] = pair_i[ss_pair_mask]
            pair_indices[:n_ss, 1] = pair_j[ss_pair_mask]
            pair_indices[n_ss:, 0] = pair_i[ds_pair_mask]
            pair_indices[n_ss:, 1] = pair_j[ds_pair_mask]
        else:
            ss_mask = x_id[:, np.newaxis] == x_id
            upper_triangle_mask = np.triu(
                np.ones_like(similarities, dtype=bool), k=1
            )
            ss = similarities[ss_mask & upper_triangle_mask]
            ds = similarities[~ss_mask & upper_triangle_mask]
            pair_indices = None
    if Z is not None:  # compute scores of X vs Z
        if Z.ndim != 2:
            raise ValueError(f"Z must be a 2D array; got ndim={Z.ndim}.")
        if z_id is None:
            raise ValueError("z_id is required when Z is provided.")
        if Z.shape[0] != len(z_id):
            raise ValueError(
                f"z_id length must match Z rows; got {len(z_id)} labels for {Z.shape[0]} rows."
            )
        similarities = cosine_similarity(X, Z)
        ss_mask = x_id[:, np.newaxis] == z_id
        ss = similarities[ss_mask]
        ds_mask = ~ss_mask
        ds = similarities[ds_mask]
        if return_pair_indices:
            ss_i, ss_j = np.nonzero(ss_mask)
            ds_i, ds_j = np.nonzero(ds_mask)
            pair_indices = np.empty((ss_i.size + ds_i.size, 2), dtype=np.int32)
            pair_indices[: ss_i.size, 0] = ss_i
            pair_indices[: ss_i.size, 1] = ss_j
            pair_indices[ss_i.size :, 0] = ds_i
            pair_indices[ss_i.size :, 1] = ds_j
        else:
            pair_indices = None

    scores = np.concatenate([ss, ds])
    y = np.concatenate([np.ones(len(ss)), np.zeros(len(ds))])
    return scores, y, pair_indices

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

DEFAULT_KEYPOINT_COLORS = ("green", "red", "green", "green", "green")


def annotate_img_with_kps(
    bgr_img: np.ndarray,
    kps: np.ndarray,
    colors: tuple[str, str, str, str, str] = DEFAULT_KEYPOINT_COLORS,
    radius: int = 2,
) -> np.ndarray:
    """
    Annotate an image with keypoints.

    Parameters:
    bgr_img (numpy.ndarray): The input image in BGR format.
    kps (numpy.ndarray): A numpy array of shape (5, 2) containing the keypoints.
    colors (tuple[str, str, str, str, str], optional): The color of each
        keypoint. By default, keypoint index 1 is red and the others are green.
        Options are 'red', 'blue', 'green', 'white', 'black'.
    radius (int, optional): The radius of the keypoints. Default is 2.

    Returns:
    numpy.ndarray: The image with keypoints annotated.
    """
    import cv2

    color_values = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }

    if kps.shape != (5, 2):
        raise ValueError(f"kps must have shape (5, 2); got {kps.shape}.")
    if len(colors) != 5:
        raise ValueError(f"colors must have length 5; got {len(colors)}.")
    invalid_colors = [color for color in colors if color not in color_values]
    if invalid_colors:
        raise ValueError(
            f"colors must contain only {sorted(color_values)}; "
            f"got invalid colors {invalid_colors}."
        )

    bgr_img_with_kps = bgr_img.copy()

    for (x, y), color in zip(kps, colors):
        cv2.circle(
            bgr_img_with_kps,
            (int(x), int(y)),
            radius=radius,
            color=color_values[color],
            thickness=-1,
        )

    return bgr_img_with_kps
