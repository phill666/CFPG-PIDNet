import numpy as np

def bootstrap_arrays(ours, other, draws=10000):
    assert ours.shape == other.shape
    (seeds, images, _) = ours.shape
    rng = np.random.default_rng(20261123)

    def estimate(value):
        counts = value[..., :3].sum(axis=-2)
        iou = counts[..., 0] / np.maximum(counts.sum(axis=-1), 1)
        geometry = value[..., 3:].mean(axis=-2)
        return np.concatenate([np.mean(iou, axis=-1)[..., None], np.mean(geometry, axis=-2)], axis=-1)
    point = estimate(ours) - estimate(other)
    values = []
    for start in range(0, draws, 100):
        size = min(100, draws - start)
        selected_seeds = rng.integers(seeds, size=(size, seeds))
        selected_images = rng.integers(images, size=(size, images))
        first = ours[selected_seeds[:, :, None], selected_images[:, None, :], :]
        second = other[selected_seeds[:, :, None], selected_images[:, None, :], :]
        values.append(estimate(first) - estimate(second))
    sampled = np.concatenate(values)
    return (point, np.percentile(sampled, 2.5, axis=0), np.percentile(sampled, 97.5, axis=0))
