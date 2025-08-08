import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import cv2 as cv
from tqdm import tqdm

try:
    import requests
except ImportError:  
    requests = None  


PROTOTXT_URLS = [
    # Author repo (caffe branch, correct path)
    "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
    # OpenCV mirrors
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/colorization_deploy_v2.prototxt",
    "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/colorization_deploy_v2.prototxt",
    # Legacy author mirror
    "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt",
]

PTS_IN_HULL_URLS = [
    # Author repo (caffe branch)
    "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy",
    # OpenCV mirrors
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/pts_in_hull.npy",
    "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/pts_in_hull.npy",
    # Legacy author mirror
    "https://raw.githubusercontent.com/richzhang/colorization/master/resources/pts_in_hull.npy",
]




def write_bytes_atomic(path: Path, content: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def http_get(url: str, desc: str) -> bytes:
    if requests is None:
        raise RuntimeError(
            f"Missing 'requests' package to download {desc}. Please 'pip install -r requirements.txt' and rerun.")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        chunks = []
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                chunks.append(chunk)
        return b"".join(chunks)


def ensure_file(path: Path, urls: Iterable[str], desc: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    last_err: Exception | None = None
    for url in urls:
        try:
            data = http_get(url, desc)
            write_bytes_atomic(path, data)
            return path
        except Exception as e:  
            last_err = e
            continue
    raise RuntimeError(f"Failed to download {desc} to {path}. Last error: {last_err}")


def ensure_proto_pts(model_dir: Path) -> Tuple[Path, Path]:
    """Ensure prototxt and pts_in_hull exist in model_dir; download if needed."""
    proto = model_dir / "colorization_deploy_v2.prototxt"
    pts = model_dir / "pts_in_hull.npy"
    ensure_file(proto, PROTOTXT_URLS, "deploy prototxt")
    ensure_file(pts, PTS_IN_HULL_URLS, "pts_in_hull.npy")
    return proto, pts


def load_net(proto: Path, caffemodel: Path, pts_in_hull: Path) -> cv.dnn_Net:
    net = cv.dnn.readNetFromCaffe(str(proto), str(caffemodel))
    pts = np.load(str(pts_in_hull))
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net


def colorize_bgr(net: cv.dnn_Net, bgr_uint8: np.ndarray) -> np.ndarray:
    bgr = (bgr_uint8.astype("float32") / 255.0)
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2Lab)
    L = lab[:, :, 0]
    L_rs = cv.resize(L, (224, 224))
    L_rs -= 50  # mean-centering
    net.setInput(cv.dnn.blobFromImage(L_rs))
    ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_us = cv.resize(ab_dec, (bgr.shape[1], bgr.shape[0]))
    lab_out = np.concatenate((L[:, :, np.newaxis], ab_us), axis=2)
    bgr_out = np.clip(cv.cvtColor(lab_out, cv.COLOR_Lab2BGR), 0, 1)
    return (bgr_out * 255).astype("uint8")


def valid_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch colorize grayscale images in a folder.")
    parser.add_argument("--input_dir", type=Path, default=Path("NonColorImg"),
                        help="Folder with grayscale images")
    parser.add_argument("--output_dir", type=Path, default=Path("ColorizedOut"),
                        help="Destination folder for colorized outputs")
    parser.add_argument("--model_dir", type=Path, default=Path("models"),
                        help="Directory to cache/download model files")
    parser.add_argument("--norebal", action="store_true",
                        help="If you specify --caffemodel, this flag is ignored; kept for backward-compat")
    parser.add_argument("--caffemodel", type=Path, default=None,
                        help="Path to local colorization .caffemodel (required; no auto-download)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir
    model_dir: Path = args.model_dir

    if not in_dir.exists():
        print(f"Input directory not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        proto, pts = ensure_proto_pts(model_dir)
    except Exception as e:  
        print(f"Error ensuring prototxt/pts: {e}", file=sys.stderr)
        sys.exit(2)

    # Resolve caffemodel locally
    caffemodel: Path | None = args.caffemodel
    if caffemodel is None:
        # Try find any caffemodel in model_dir
        candidates = sorted([p for p in model_dir.glob("*.caffemodel")], key=lambda p: p.name)
        if candidates:
            # Prefer colorization_release_v2*.caffemodel
            preferred = [p for p in candidates if p.name.startswith("colorization_release_v2")]
            caffemodel = preferred[0] if preferred else candidates[0]
        else:
            print(
                "Missing .caffemodel. Please place your caffemodel into 'models' or pass --caffemodel <path>.\n"
                "Example filenames: colorization_release_v2.caffemodel (rebalanced) or colorization_release_v2_norebal.caffemodel",
                file=sys.stderr)
            sys.exit(3)

    if not caffemodel.exists():
        print(f"Specified caffemodel not found: {caffemodel}", file=sys.stderr)
        sys.exit(3)

    net = load_net(proto, caffemodel, pts)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in in_dir.iterdir() if p.is_file() and valid_image(p)]
    if not images:
        print(f"No images found in {in_dir}")
        return

    for img_path in tqdm(images, desc="Colorizing", unit="img"):
        dst = out_dir / img_path.name
        if dst.exists() and not args.overwrite:
            continue
        bgr = cv.imread(str(img_path), cv.IMREAD_COLOR)
        if bgr is None:
            continue
        out = colorize_bgr(net, bgr)
        cv.imwrite(str(dst), out)

    print(f"Done. Saved to: {out_dir}")


if __name__ == "__main__":
    main()

