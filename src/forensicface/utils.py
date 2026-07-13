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
    return_pair_indices: bool = False,
    block_size: int = 2048,
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
        block_size: Maximum number of rows and columns in each similarity tile.
            Smaller values reduce peak working memory without changing the
            returned arrays.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray | None]: Scores, Boolean
        same-source labels, and optional ``int32`` pair indices. ``True`` means
        same-source and ``False`` means different-source. For ``X`` vs ``X``,
        both index columns refer to rows in ``X``. For ``X`` vs ``Z``, the first
        column refers to ``X`` and the second column refers to ``Z``.

        Same-source scores precede different-source scores. Ordering within
        those two sections is unspecified.
    """
    X = np.asarray(X)
    x_id = np.asarray(x_id)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array; got ndim={X.ndim}.")
    if X.shape[0] != len(x_id):
        raise ValueError(
            f"x_id length must match X rows; got {len(x_id)} labels for {X.shape[0]} rows."
        )
    if not isinstance(block_size, (int, np.integer)) or block_size <= 0:
        raise ValueError(f"block_size must be a positive integer; got {block_size!r}.")
    block_size = int(block_size)

    if Z is not None:
        Z = np.asarray(Z)
        if Z.ndim != 2:
            raise ValueError(f"Z must be a 2D array; got ndim={Z.ndim}.")
        if z_id is None:
            raise ValueError("z_id is required when Z is provided.")
        z_id = np.asarray(z_id)
        if Z.shape[0] != len(z_id):
            raise ValueError(
                f"z_id length must match Z rows; got {len(z_id)} labels for {Z.shape[0]} rows."
            )
        if X.shape[1] != Z.shape[1]:
            raise ValueError(
                "X and Z must have the same embedding dimension; "
                f"got {X.shape[1]} and {Z.shape[1]}."
            )

    # Normalize once, instead of normalizing X twice in the X-vs-X case.
    normalized_X = X / np.linalg.norm(X, axis=1, keepdims=True)
    normalized_Z = (
        normalized_X
        if Z is None
        else Z / np.linalg.norm(Z, axis=1, keepdims=True)
    )

    if Z is None:
        n_pairs = X.shape[0] * (X.shape[0] - 1) // 2
        _, identity_counts = np.unique(x_id, return_counts=True)
        n_ss = int(np.sum(identity_counts * (identity_counts - 1) // 2))
    else:
        n_pairs = X.shape[0] * Z.shape[0]
        x_values, x_counts = np.unique(x_id, return_counts=True)
        z_values, z_counts = np.unique(z_id, return_counts=True)
        _, x_common, z_common = np.intersect1d(
            x_values,
            z_values,
            assume_unique=True,
            return_indices=True,
        )
        n_ss = int(np.sum(x_counts[x_common] * z_counts[z_common]))

    score_dtype = np.result_type(normalized_X.dtype, normalized_Z.dtype)
    scores = np.empty(n_pairs, dtype=score_dtype)
    y = np.zeros(n_pairs, dtype=bool)
    y[:n_ss] = True
    pair_indices = (
        np.empty((n_pairs, 2), dtype=np.int32)
        if return_pair_indices
        else None
    )

    ss_position = 0
    ds_position = n_ss
    z_rows = normalized_Z.shape[0]

    for x_start in range(0, normalized_X.shape[0], block_size):
        x_stop = min(x_start + block_size, normalized_X.shape[0])
        z_first = x_start if Z is None else 0

        for z_start in range(z_first, z_rows, block_size):
            z_stop = min(z_start + block_size, z_rows)
            similarities = (
                normalized_X[x_start:x_stop]
                @ normalized_Z[z_start:z_stop].T
            )
            same_mask = (
                x_id[x_start:x_stop, np.newaxis]
                == (x_id if Z is None else z_id)[z_start:z_stop]
            )

            if Z is None and x_start == z_start:
                eligible_mask = np.triu(
                    np.ones(similarities.shape, dtype=bool),
                    k=1,
                )
                same_mask &= eligible_mask
                different_mask = eligible_mask & ~same_mask
            else:
                different_mask = ~same_mask

            n_tile_ss = int(np.count_nonzero(same_mask))
            n_tile_ds = int(np.count_nonzero(different_mask))
            next_ss_position = ss_position + n_tile_ss
            next_ds_position = ds_position + n_tile_ds

            scores[ss_position:next_ss_position] = similarities[same_mask]
            scores[ds_position:next_ds_position] = similarities[different_mask]

            if pair_indices is not None:
                ss_i, ss_j = np.nonzero(same_mask)
                ds_i, ds_j = np.nonzero(different_mask)
                pair_indices[ss_position:next_ss_position, 0] = ss_i + x_start
                pair_indices[ss_position:next_ss_position, 1] = ss_j + z_start
                pair_indices[ds_position:next_ds_position, 0] = ds_i + x_start
                pair_indices[ds_position:next_ds_position, 1] = ds_j + z_start

            ss_position = next_ss_position
            ds_position = next_ds_position

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
