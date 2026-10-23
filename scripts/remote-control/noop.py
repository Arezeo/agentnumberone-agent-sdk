"""Copy this file and pass --policy /path/to/policy.py to use your own model."""


def decide(frame, controls):
    # frame is a decoded av.VideoFrame. Pixels: frame.to_ndarray(format="rgb24").
    # Inference runs on a worker thread. Video reception continues in parallel.
    # Only call controls.* here; returning does not guarantee action execution.
    controls.releaseAll()
