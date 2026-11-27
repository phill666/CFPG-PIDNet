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


IMAGE_PRIORITY = [".jpg", ".jpeg", ".bmp", ".png"]


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


def _choose_image_path(paths_by_suffix):
    for suffix in IMAGE_PRIORITY:
        if suffix in paths_by_suffix:
            return paths_by_suffix[suffix]
    return None


def scan_mtd_class(root, class_name, image_subdir="Imgs", free_class="MT_Free"):
    class_dir = _resolve_child(_resolve_child(root, class_name), image_subdir)
    if not class_dir.exists():
        raise FileNotFoundError("Magnetic Tile class image directory does not exist: {}".format(class_dir))

    files_by_stem = {}
    duplicate_suffixes = []
    for path in class_dir.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in IMAGE_PRIORITY:
            continue
        stem = path.stem
        files_by_stem.setdefault(stem, {})
        if suffix in files_by_stem[stem]:
            duplicate_suffixes.append("{}{}".format(stem, suffix))
        files_by_stem[stem][suffix] = path

    if duplicate_suffixes:
        raise ValueError(
            "Duplicate Magnetic Tile files in {}: {}".format(class_dir, sorted(set(duplicate_suffixes))[:20])
        )

    is_free = str(class_name) == str(free_class)
    samples = {}
    missing_masks = []
    zero_mask_count = 0
    mask_count = 0
    image_count = 0

    for stem in sorted(files_by_stem):
        suffix_paths = files_by_stem[stem]
        image_path = _choose_image_path(suffix_paths)
        if image_path is None:
            continue
        image_count += 1

        # When jpg/jpeg/bmp and png share a stem, the non-png file is the image
        # and the png is the mask. A png-only sample is treated as an image.
        mask_path = suffix_paths.get(".png") if image_path.suffix.lower() != ".png" else None
        if mask_path is None:
            if is_free:
                zero_mask_count += 1
            else:
                missing_masks.append(stem)
                continue
        else:
            mask_count += 1

        entry = "{}/{}".format(class_name, stem)
        samples[entry] = {
            "entry": entry,
            "class_name": class_name,
            "stem": stem,
            "image": image_path,
            "mask": mask_path,
            "zero_mask": mask_path is None,
        }

    if missing_masks:
        preview = ", ".join(missing_masks[:30])
        if len(missing_masks) > 30:
            preview += ", ..."
        raise FileNotFoundError(
            "Magnetic Tile defect class {} images missing paired png masks: {}".format(class_name, preview)
        )

    stats = {
        "class_dir": str(class_dir),
        "image_count": image_count,
        "mask_count": mask_count,
        "zero_mask_count": zero_mask_count,
        "sample_count": len(samples),
        "missing_masks": missing_masks,
    }
    return samples, stats


def scan_mtd_pairs(root, classes, image_subdir="Imgs", free_class="MT_Free", verbose=True):
    root = Path(root)
    all_samples = {}
    class_stats = {}
    for class_name in classes:
        class_samples, stats = scan_mtd_class(root, class_name, image_subdir=image_subdir, free_class=free_class)
        overlap = sorted(set(all_samples) & set(class_samples))
        if overlap:
            raise ValueError("Duplicate Magnetic Tile split entries across classes: {}".format(overlap[:20]))
        all_samples.update(class_samples)
        class_stats[class_name] = stats
        if verbose:
            print(
                "Magnetic Tile pairing: {} images={}, masks={}, zero_masks={}, samples={}".format(
                    class_name,
                    stats["image_count"],
                    stats["mask_count"],
                    stats["zero_mask_count"],
                    stats["sample_count"],
                )
            )

    return all_samples, {"classes": class_stats, "total_pairs": len(all_samples)}


def _split_class_entries(entries, train_ratio, val_ratio, seed):
    entries = list(entries)
    rng = random.Random(int(seed))
    rng.shuffle(entries)
    n_total = len(entries)
    n_train = int(n_total * float(train_ratio))
    n_val = int(n_total * float(val_ratio))
    n_test = n_total - n_train - n_val

    if n_total >= 3:
        if n_train <= 0:
            n_train = 1
        if n_val <= 0:
            n_val = 1
        if n_test <= 0:
            n_test = 1
        while n_train + n_val + n_test > n_total:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

    train_entries = entries[:n_train]
    val_entries = entries[n_train:n_train + n_val]
    test_entries = entries[n_train + n_val:]
    return train_entries, val_entries, test_entries


def prepare_mtd_splits(
    root,
    classes,
    image_subdir,
    split_dir,
    train_split="mtd_train.txt",
    val_split="mtd_val.txt",
    test_split="mtd_test.txt",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    rebuild_split=False,
    free_class="MT_Free",
    verbose=True,
):
    samples, stats = scan_mtd_pairs(root, classes, image_subdir=image_subdir, free_class=free_class, verbose=verbose)
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
            print("Using existing Magnetic Tile splits from {}".format(split_root))
        return splits, split_paths, stats

    total_ratio = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Magnetic Tile train_ratio + val_ratio + test_ratio must be 1.0")

    splits = {"train": [], "val": [], "test": []}
    for class_name in classes:
        class_entries = sorted(entry for entry, sample in samples.items() if sample["class_name"] == class_name)
        train_entries, val_entries, test_entries = _split_class_entries(
            class_entries,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=int(seed),
        )
        splits["train"].extend(train_entries)
        splits["val"].extend(val_entries)
        splits["test"].extend(test_entries)

    for name in splits:
        splits[name] = sorted(splits[name])
        _write_lines(split_paths[name], splits[name])

    if verbose:
        print(
            "Generated Magnetic Tile splits at {}: train={}, val={}, test={}".format(
                split_root, len(splits["train"]), len(splits["val"]), len(splits["test"])
            )
        )
    return splits, split_paths, stats


class MagneticTileDataset(Dataset):
    def __init__(
        self,
        root,
        classes,
        image_subdir="Imgs",
        split_file=None,
        entries=None,
        split_name="train",
        input_size=512,
        is_train=True,
        free_class="MT_Free",
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
        self.classes = list(classes)
        self.image_subdir = image_subdir
        self.split_name = str(split_name)
        self.input_size = int(input_size)
        self.is_train = bool(is_train)
        self.free_class = free_class
        self.scale_range = tuple(scale_range)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotation_degrees = float(rotation_degrees)
        self.color_jitter = tuple(color_jitter)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

        samples, _ = scan_mtd_pairs(
            root,
            self.classes,
            image_subdir=self.image_subdir,
            free_class=self.free_class,
            verbose=verbose,
        )
        if split_file is not None:
            entries = _load_lines(_resolve_repo_path(split_file))
        if entries is None:
            entries = sorted(samples.keys())

        missing = [entry for entry in entries if entry not in samples]
        if missing:
            preview = ", ".join(missing[:30])
            if len(missing) > 30:
                preview += ", ..."
            raise FileNotFoundError(
                "Magnetic Tile split {} references missing samples: {}".format(self.split_name, preview)
            )
        self.samples = [samples[entry] for entry in entries]

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, sample):
        image = Image.open(sample["image"]).convert("RGB")
        if sample["zero_mask"]:
            mask = Image.new("L", image.size, 0)
        else:
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
        return image_tensor, mask_tensor, sample["entry"]
