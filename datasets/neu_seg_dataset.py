from pathlib import Path
import random
import warnings

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset


try:
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
except AttributeError:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


def _path_from_config(value):
    path = Path(value)
    if path.is_absolute():
        return path
    parts = [part for part in str(value).replace("\\", "/").split("/") if part]
    return Path(*parts) if parts else Path(".")


def _resolve_child(root, child):
    child_path = _path_from_config(child)
    return child_path if child_path.is_absolute() else Path(root) / child_path


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(path):
    path = _path_from_config(path)
    return path if path.is_absolute() else _repo_root() / path


def _load_lines(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_lines(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write("{}\n".format(value))


def _files_by_stem(directory, suffix):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError("NEU Seg directory does not exist: {}".format(directory))

    files = {}
    duplicates = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != suffix:
            continue
        if path.stem in files:
            duplicates.append(path.stem)
        files[path.stem] = path

    if duplicates:
        raise ValueError(
            "Duplicate NEU Seg {} stems found in {}: {}".format(
                suffix, directory, sorted(set(duplicates))[:20]
            )
        )
    return files


def scan_neu_seg_pairs(root, image_dir, mask_dir, verbose=True):
    root = Path(root)
    image_root = _resolve_child(root, image_dir)
    mask_root = _resolve_child(root, mask_dir)

    image_files = _files_by_stem(image_root, ".jpg")
    mask_files = _files_by_stem(mask_root, ".png")
    image_stems = set(image_files)
    mask_stems = set(mask_files)

    missing_masks = sorted(image_stems - mask_stems)
    if missing_masks:
        preview = ", ".join(missing_masks[:30])
        if len(missing_masks) > 30:
            preview += ", ..."
        raise FileNotFoundError(
            "NEU Seg images missing paired png masks: {}".format(preview)
        )

    missing_images = sorted(mask_stems - image_stems)
    if missing_images:
        preview = ", ".join(missing_images[:30])
        if len(missing_images) > 30:
            preview += ", ..."
        warnings.warn(
            "NEU Seg png masks missing paired jpg images: {}".format(preview),
            RuntimeWarning,
        )

    if len(image_files) != len(mask_files):
        warnings.warn(
            "NEU Seg image/mask counts differ: images={}, masks={}".format(
                len(image_files), len(mask_files)
            ),
            RuntimeWarning,
        )

    common_stems = sorted(image_stems & mask_stems)
    pairs = {
        stem: {"stem": stem, "image": image_files[stem], "mask": mask_files[stem]}
        for stem in common_stems
    }
    stats = {
        "image_count": len(image_files),
        "mask_count": len(mask_files),
        "pair_count": len(pairs),
        "missing_masks": missing_masks,
        "missing_images": missing_images,
        "image_root": str(image_root),
        "mask_root": str(mask_root),
    }
    if verbose:
        print(
            "NEU Seg pairing: images={}, masks={}, pairs={}".format(
                stats["image_count"], stats["mask_count"], stats["pair_count"]
            )
        )
    return pairs, stats


def prepare_neu_seg_splits(
    root,
    image_dir,
    mask_dir,
    split_dir,
    train_split="neu_seg_train.txt",
    val_split="neu_seg_val.txt",
    test_split="neu_seg_test.txt",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    rebuild_split=False,
    verbose=True,
):
    pairs, stats = scan_neu_seg_pairs(root, image_dir, mask_dir, verbose=verbose)
    split_root = _resolve_repo_path(split_dir)
    split_paths = {
        "train": split_root / train_split,
        "val": split_root / val_split,
        "test": split_root / test_split,
    }

    have_all_splits = all(path.exists() for path in split_paths.values())
    if have_all_splits and not rebuild_split:
        splits = {name: _load_lines(path) for name, path in split_paths.items()}
        if verbose:
            print("Using existing NEU Seg splits from {}".format(split_root))
        return splits, split_paths, stats

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("NEU Seg split ratios must sum to 1.0")

    stems = sorted(pairs.keys())
    rng = random.Random(int(seed))
    rng.shuffle(stems)

    total = len(stems)
    n_train = int(total * float(train_ratio))
    n_val = int(total * float(val_ratio))
    splits = {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }

    for name, path in split_paths.items():
        _write_lines(path, splits[name])

    if verbose:
        print(
            "Generated NEU Seg splits at {}: train={}, val={}, test={}".format(
                split_root, len(splits["train"]), len(splits["val"]), len(splits["test"])
            )
        )
    return splits, split_paths, stats


class NEUSegDataset(Dataset):
    def __init__(
        self,
        root,
        image_dir,
        mask_dir,
        split_file=None,
        stems=None,
        split_name="train",
        input_size=512,
        is_train=True,
        scale_range=(0.75, 1.5),
        hflip_prob=0.5,
        vflip_prob=0.5,
        rotation_degrees=10.0,
        color_jitter=(0.1, 0.1),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        verbose=False,
    ):
        self.root = Path(root)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.split_name = split_name
        self.input_size = int(input_size)
        self.is_train = bool(is_train)
        self.scale_range = tuple(scale_range)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotation_degrees = float(rotation_degrees)
        self.color_jitter = tuple(color_jitter)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

        pairs, _ = scan_neu_seg_pairs(root, image_dir, mask_dir, verbose=verbose)
        if split_file is not None:
            split_path = _resolve_repo_path(split_file)
            stems = _load_lines(split_path)
        if stems is None:
            stems = sorted(pairs.keys())

        missing = [stem for stem in stems if stem not in pairs]
        if missing:
            preview = ", ".join(missing[:30])
            if len(missing) > 30:
                preview += ", ..."
            raise FileNotFoundError(
                "NEU Seg split {} references missing paired samples: {}".format(
                    self.split_name, preview
                )
            )

        self.samples = [pairs[stem] for stem in stems]

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, sample):
        image = Image.open(sample["image"]).convert("RGB")
        mask = Image.open(sample["mask"]).convert("L")
        mask = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
        return image, mask

    def _resize(self, image, mask, size):
        return (
            image.resize((size, size), BILINEAR),
            mask.resize((size, size), NEAREST),
        )

    def _random_scale(self, image, mask):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        width, height = image.size
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return image.resize(new_size, BILINEAR), mask.resize(new_size, NEAREST)

    def _random_crop(self, image, mask):
        crop = self.input_size
        width, height = image.size
        pad_w = max(crop - width, 0)
        pad_h = max(crop - height, 0)
        if pad_w > 0 or pad_h > 0:
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=(0, 0, 0))
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)
            width, height = image.size

        if width == crop:
            left = 0
        else:
            left = random.randint(0, width - crop)
        if height == crop:
            top = 0
        else:
            top = random.randint(0, height - crop)
        box = (left, top, left + crop, top + crop)
        return image.crop(box), mask.crop(box)

    def _random_rotate(self, image, mask):
        if self.rotation_degrees <= 0:
            return image, mask
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        image = image.rotate(angle, resample=BILINEAR, fillcolor=(0, 0, 0))
        mask = mask.rotate(angle, resample=NEAREST, fillcolor=0)
        return image, mask

    def _color_jitter(self, image):
        brightness, contrast = self.color_jitter
        if brightness > 0:
            factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if contrast > 0:
            factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image

    def _augment_train(self, image, mask):
        image, mask = self._random_scale(image, mask)
        image, mask = self._random_crop(image, mask)
        if random.random() < self.hflip_prob:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if random.random() < self.vflip_prob:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)
        image, mask = self._random_rotate(image, mask)
        image = self._color_jitter(image)
        return image, mask

    def _to_tensors(self, image, mask):
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - self.mean) / self.std
        image_np = image_np.transpose(2, 0, 1).copy()
        mask_np = (np.asarray(mask) > 0).astype(np.int64)
        return torch.from_numpy(image_np).float(), torch.from_numpy(mask_np).long()

    def __getitem__(self, index):
        sample = self.samples[index]
        image, mask = self._load_sample(sample)
        if self.is_train:
            image, mask = self._augment_train(image, mask)
        else:
            image, mask = self._resize(image, mask, self.input_size)

        image_tensor, mask_tensor = self._to_tensors(image, mask)
        return image_tensor, mask_tensor, sample["stem"]
