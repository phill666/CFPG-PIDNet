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


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _path_from_config(value):
    path = Path(value)
    if path.is_absolute():
        return path
    parts = [part for part in str(value).replace("\\", "/").split("/") if part]
    return Path(*parts) if parts else Path(".")


def _resolve_repo_path(path):
    path = _path_from_config(path)
    return path if path.is_absolute() else _repo_root() / path


def _resolve_child(root, child):
    child = _path_from_config(child)
    return child if child.is_absolute() else Path(root) / child


def _load_lines(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_lines(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write("{}\n".format(value))


def _mask_to_image_stem(mask_stem):
    return str(mask_stem).replace("_GT", "", 1)


def scan_ksdd2_split(split_dir, split_name):
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError("KolektorSDD2 split directory does not exist: {}".format(split_dir))

    image_files = {}
    mask_files = {}
    duplicate_images = []
    duplicate_masks = []

    for path in split_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        if "_GT" in path.stem:
            stem = _mask_to_image_stem(path.stem)
            if stem in mask_files:
                duplicate_masks.append(stem)
            mask_files[stem] = path
        else:
            stem = path.stem
            if stem in image_files:
                duplicate_images.append(stem)
            image_files[stem] = path

    if duplicate_images:
        raise ValueError(
            "Duplicate KSDD2 image stems in {}: {}".format(split_dir, sorted(set(duplicate_images))[:20])
        )
    if duplicate_masks:
        raise ValueError(
            "Duplicate KSDD2 mask stems in {}: {}".format(split_dir, sorted(set(duplicate_masks))[:20])
        )

    image_stems = set(image_files)
    mask_stems = set(mask_files)
    missing_masks = sorted(image_stems - mask_stems)
    if missing_masks:
        preview = ", ".join(missing_masks[:30])
        if len(missing_masks) > 30:
            preview += ", ..."
        raise FileNotFoundError(
            "KolektorSDD2 {} images missing paired GT masks: {}".format(split_name, preview)
        )

    missing_images = sorted(mask_stems - image_stems)
    if missing_images:
        preview = ", ".join(missing_images[:30])
        if len(missing_images) > 30:
            preview += ", ..."
        warnings.warn(
            "KolektorSDD2 {} GT masks missing paired images: {}".format(split_name, preview),
            RuntimeWarning,
        )

    if len(image_files) != len(mask_files):
        warnings.warn(
            "KolektorSDD2 {} image/mask counts differ: images={}, masks={}".format(
                split_name, len(image_files), len(mask_files)
            ),
            RuntimeWarning,
        )

    stems = sorted(image_stems & mask_stems)
    pairs = {
        stem: {"stem": stem, "image": image_files[stem], "mask": mask_files[stem], "source": split_name}
        for stem in stems
    }
    stats = {
        "image_count": len(image_files),
        "mask_count": len(mask_files),
        "pair_count": len(pairs),
        "missing_masks": missing_masks,
        "missing_images": missing_images,
        "split_dir": str(split_dir),
    }
    return pairs, stats


def scan_ksdd2_pairs(root, train_dir, test_dir, verbose=True):
    root = Path(root)
    train_pairs, train_stats = scan_ksdd2_split(_resolve_child(root, train_dir), "train")
    test_pairs, test_stats = scan_ksdd2_split(_resolve_child(root, test_dir), "test")
    if verbose:
        print(
            "KolektorSDD2 pairing: train images={}, masks={}, pairs={}; test images={}, masks={}, pairs={}".format(
                train_stats["image_count"],
                train_stats["mask_count"],
                train_stats["pair_count"],
                test_stats["image_count"],
                test_stats["mask_count"],
                test_stats["pair_count"],
            )
        )
    return {"train": train_pairs, "test": test_pairs}, {"train": train_stats, "test": test_stats}


def prepare_ksdd2_splits(
    root,
    train_dir,
    test_dir,
    split_dir,
    train_split="ksdd2_train.txt",
    val_split="ksdd2_val.txt",
    test_split="ksdd2_test.txt",
    train_ratio=0.85,
    val_ratio=0.15,
    seed=42,
    rebuild_split=False,
    verbose=True,
):
    pairs, stats = scan_ksdd2_pairs(root, train_dir, test_dir, verbose=verbose)
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
            print("Using existing KolektorSDD2 splits from {}".format(split_root))
        return splits, split_paths, stats

    if abs((float(train_ratio) + float(val_ratio)) - 1.0) > 1e-6:
        raise ValueError("KolektorSDD2 train_ratio + val_ratio must be 1.0")

    train_stems = sorted(pairs["train"].keys())
    rng = random.Random(int(seed))
    rng.shuffle(train_stems)
    n_train = int(len(train_stems) * float(train_ratio))
    splits = {
        "train": train_stems[:n_train],
        "val": train_stems[n_train:],
        "test": sorted(pairs["test"].keys()),
    }
    for name, path in split_paths.items():
        _write_lines(path, splits[name])
    if verbose:
        print(
            "Generated KolektorSDD2 splits at {}: train={}, val={}, test={}".format(
                split_root, len(splits["train"]), len(splits["val"]), len(splits["test"])
            )
        )
    return splits, split_paths, stats


class KolektorSDD2Dataset(Dataset):
    def __init__(
        self,
        root,
        train_dir,
        test_dir,
        split_file=None,
        stems=None,
        split_name="train",
        input_size=512,
        is_train=True,
        scale_range=(0.75, 1.5),
        hflip_prob=0.5,
        vflip_prob=0.5,
        rotation_degrees=0.0,
        color_jitter=(0.1, 0.1),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        verbose=False,
    ):
        self.root = Path(root)
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.split_name = str(split_name)
        self.input_size = int(input_size)
        self.is_train = bool(is_train)
        self.scale_range = tuple(scale_range)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotation_degrees = float(rotation_degrees)
        self.color_jitter = tuple(color_jitter)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

        pairs, _ = scan_ksdd2_pairs(root, train_dir, test_dir, verbose=verbose)
        source = "test" if self.split_name.lower() == "test" else "train"
        source_pairs = pairs[source]
        if split_file is not None:
            stems = _load_lines(_resolve_repo_path(split_file))
        if stems is None:
            stems = sorted(source_pairs.keys())

        missing = [stem for stem in stems if stem not in source_pairs]
        if missing:
            preview = ", ".join(missing[:30])
            if len(missing) > 30:
                preview += ", ..."
            raise FileNotFoundError(
                "KolektorSDD2 split {} references missing {} samples: {}".format(
                    self.split_name, source, preview
                )
            )
        self.samples = [source_pairs[stem] for stem in stems]

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, sample):
        image = Image.open(sample["image"]).convert("RGB")
        mask = Image.open(sample["mask"]).convert("L")
        mask = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
        return image, mask

    def _resize(self, image, mask, size):
        return image.resize((size, size), BILINEAR), mask.resize((size, size), NEAREST)

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
        left = 0 if width == crop else random.randint(0, width - crop)
        top = 0 if height == crop else random.randint(0, height - crop)
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
            image = ImageEnhance.Brightness(image).enhance(random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness))
        if contrast > 0:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast))
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
