"""Contact sheet: several observations in one image.

A 2 Hz band sees the game as a handful of stills; a fast vision model should
see them *together*, in one encode, rather than as N separate images. This
module tiles recent frames into a single labelled grid:

    +----------------+----------------+----------------+
    | #12  t-1.5s    | #13  t-1.0s    | #14  t-0.5s    |
    | [W]            | [W]            | [A]            |
    +----------------+----------------+----------------+
    | #15  t-0.0s (NOW)                 |  (empty)        |
    +----------------+----------------+----------------+

Reading order is left-to-right, top-to-bottom, oldest first, so the last tile
is always the current frame. Each tile carries its sequence number, age and
the action that was dispatched after it -- without that, a grid of near
identical road frames is ambiguous about what the agent did between them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - Pillow is optional in the headless venv
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore


def to_rgb(frame) -> np.ndarray:
    if hasattr(frame, "to_ndarray"):
        frame = frame.to_ndarray(format="rgb24")
    if Image is not None and isinstance(frame, Image.Image):
        frame = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


@dataclass
class Tile:
    seq: int
    age_ms: int
    action: str = ""      # what the agent did after this frame, e.g. "W"
    label: str = ""       # free-form caption (e.g. "dead" or task id)

    def caption(self) -> str:
        parts = [f"#{self.seq}", f"{self.age_ms / 1000:.1f}s"]
        if self.label:
            parts.append(self.label)
        if self.action:
            parts.append(self.action)
        return " ".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        return {"seq": self.seq, "age_ms": self.age_ms, "action": self.action, "label": self.label}


def _downscale(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest/bilinear downscale without requiring Pillow."""
    h, w = arr.shape[:2]
    if Image is not None:
        return np.asarray(Image.fromarray(arr).resize((width, height), Image.BILINEAR), dtype=np.uint8)
    ys = (np.arange(height) * (h / height)).astype(np.int32)
    xs = (np.arange(width) * (w / width)).astype(np.int32)
    return arr[ys][:, xs]


def layout(count: int) -> Tuple[int, int]:
    """Rows/cols for a sheet: 2x3 up to six tiles; 1xN beyond that."""
    if count <= 3:
        return 1, max(1, count)
    if count <= 6:
        return 2, 3
    return 1, count


def build(
    frames: Sequence[Tuple[np.ndarray, Tile]],
    *,
    tile_size: Tuple[int, int] = (320, 180),
    keep_aspect: bool = True,
    gutter: int = 4,
    caption_px: int = 18,
    crop: Optional[Any] = None,   # any object with .apply(rgb) -> rgb
    now_label: str = "NOW",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Compose a labelled contact sheet. Returns (sheet_rgb, tile metadata).

    With `keep_aspect` (default) the tile width is `tile_size[0]` and the height
    follows the frame's own shape, so a 4:3 game is not squashed into 16:9.
    """
    if not frames:
        raise ValueError("a contact sheet needs at least one frame")
    tw, th = tile_size
    if keep_aspect:
        first = to_rgb(frames[0][0])
        first = crop.apply(first) if crop is not None else first
        src_h, src_w = first.shape[:2]
        if src_w > 0 and src_h > 0:
            th = max(1, int(round(tw * src_h / src_w)))
    rows, cols = layout(len(frames))
    cell_w, cell_h = tw + gutter, th + caption_px + gutter
    sheet = np.full((rows * cell_h + gutter, cols * cell_w + gutter, 3), 24, np.uint8)
    meta: List[Dict[str, Any]] = []

    for index, (rgb, tile) in enumerate(frames):
        row, col = divmod(index, cols)
        y0 = gutter + row * cell_h + caption_px
        x0 = gutter + col * cell_w
        source = crop.apply(to_rgb(rgb)) if crop is not None else to_rgb(rgb)
        sheet[y0:y0 + th, x0:x0 + tw] = _downscale(source, tw, th)
        label = tile.caption()
        if index == len(frames) - 1 and not tile.label:
            label += " " + now_label
        _draw_caption(sheet, label, x0, y0, tw, caption_px)
        meta.append(dict(tile.as_dict(), tile=[row, col], caption=label))

    return sheet, meta


_WARNED: Dict[str, bool] = {}


def _draw_caption(sheet: np.ndarray, text: str, x0: int, y0: int, width: int, caption_px: int) -> None:
    """Text on the dark caption strip above a tile; needs Pillow to render glyphs.

    Without Pillow the tiles are still composited but carry no caption, which
    makes the sheet much less useful, so this warns once instead of failing
    silently.
    """
    strip_y0 = y0 - caption_px + 2
    sheet[strip_y0:y0 - 1, x0:x0 + width] = (16, 16, 16)
    if Image is None or ImageDraw is None:
        if not _WARNED.get("no_pillow"):
            _WARNED["no_pillow"] = True
            import sys

            print("[sheet] WARNING: Pillow is not installed, so contact-sheet tiles "
                  "carry no captions (seq/age/action). Install pillow to restore them.",
                  file=sys.stderr, flush=True)
        return
    image = Image.fromarray(sheet)
    draw = ImageDraw.Draw(image)
    draw.text((x0 + 4, strip_y0), text, fill=(235, 235, 235))
    sheet[:] = np.asarray(image, dtype=np.uint8)


def stack_vertical(images: Sequence[np.ndarray]) -> np.ndarray:
    """A one-column fallback when a grid would be too wide for the model."""
    arrays = [to_rgb(image) for image in images]
    width = min(a.shape[1] for a in arrays)
    scaled = []
    for arr in arrays:
        if arr.shape[1] != width:
            height = max(1, int(round(arr.shape[0] * width / arr.shape[1])))
            arr = _downscale(arr, width, height)
        scaled.append(arr)
    return np.concatenate(scaled, axis=0)


def encode_png(rgb: np.ndarray) -> bytes:
    """PNG bytes for an H x W x 3 uint8 array (no Pillow required)."""
    import av

    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
    codec = av.CodecContext.create("png", "w")
    codec.width, codec.height, codec.pix_fmt = frame.width, frame.height, "rgb24"
    packets = codec.encode(frame.reformat(format="rgb24")) + codec.encode(None)
    return b"".join(bytes(packet) for packet in packets)


def summary(meta: Sequence[Dict[str, Any]]) -> str:
    """One-line text description of the sheet's reading order, for the prompt."""
    return json.dumps([{k: m[k] for k in ("seq", "age_ms", "action")} for m in meta], separators=(",", ":"))
