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
]

PTS_IN_HULL_URLS = [
    # Author repo (caffe branch)
    "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy",

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


def apply_clahe_on_L(bgr_uint8: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Apply CLAHE on L channel and return enhanced BGR uint8 image."""
    if clip_limit <= 0:
        return bgr_uint8
    bgr = bgr_uint8.astype("float32") / 255.0
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2Lab)
    L = (lab[:, :, 0] * 255.0 / 100.0).astype("uint8")
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    L_enh = clahe.apply(L)
    lab[:, :, 0] = (L_enh.astype("float32") * (100.0 / 255.0))
    out = np.clip(cv.cvtColor(lab, cv.COLOR_Lab2BGR), 0, 1)
    return (out * 255).astype("uint8")


def unsharp_mask(bgr_uint8: np.ndarray, amount: float, radius: int = 1) -> np.ndarray:
    """Simple unsharp mask using Gaussian blur."""
    if amount <= 0:
        return bgr_uint8
    sigma = max(0.0, float(radius))
    blur = cv.GaussianBlur(bgr_uint8, (0, 0), sigmaX=sigma, sigmaY=sigma)
 
    out = cv.addWeighted(bgr_uint8, 1.0 + amount, blur, -amount, 0)
    return out


def ensure_sr_model(model_dir: Path, scale: int) -> Path:
    """Ensure ESPCN_x{scale}.pb model is available."""
    assert scale in (2, 4)
    url = f"https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x{scale}.pb"
    path = model_dir / f"ESPCN_x{scale}.pb"
    return ensure_file(path, [url], f"ESPCN_x{scale}.pb")


def upscale_with_espcn(bgr_uint8: np.ndarray, model_path: Path, scale: int) -> np.ndarray:
    sr = cv.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel("espcn", scale)
    return sr.upsample(bgr_uint8)


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
    # Pre-processing
    parser.add_argument("--pre_clahe", type=float, default=0.0,
                        help="Apply CLAHE on L before colorization (clipLimit, 0 disables)")
    parser.add_argument("--pre_clahe_grid", type=int, default=8,
                        help="CLAHE tileGridSize (NxN)")
    parser.add_argument("--pre_unsharp", type=float, default=0.0,
                        help="Unsharp mask amount before colorization (0 disables)")
    # Post-processing
    parser.add_argument("--post_upscale", type=int, choices=[0, 2, 4], default=0,
                        help="Upscale with ESPCN x2/x4 after colorization (0 disables)")
    parser.add_argument("--post_unsharp", type=float, default=0.0,
                        help="Unsharp mask amount after colorization (0 disables)")
    parser.add_argument("--post_denoise", type=float, default=0.0,
                        help="FastNLMeans denoise strength (0 disables)")
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
        # Pre-processing
        if args.pre_clahe > 0.0:
            bgr = apply_clahe_on_L(bgr, clip_limit=args.pre_clahe, tile_grid=args.pre_clahe_grid)
        if args.pre_unsharp > 0.0:
            bgr = unsharp_mask(bgr, amount=args.pre_unsharp, radius=1)

        out = colorize_bgr(net, bgr)

        # Post-processing
        if args.post_upscale in (2, 4):
            try:
                sr_path = ensure_sr_model(model_dir, args.post_upscale)
                out = upscale_with_espcn(out, sr_path, args.post_upscale)
            except Exception as e:  
                print(f"SR model error: {e}")
        if args.post_unsharp > 0.0:
            out = unsharp_mask(out, amount=args.post_unsharp, radius=1)
        if args.post_denoise > 0.0:
            h = float(args.post_denoise)
            out = cv.fastNlMeansDenoisingColored(out, None, h, h, 7, 21)
        cv.imwrite(str(dst), out)

    print(f"Done. Saved to: {out_dir}")


if __name__ == "__main__":
    main()

