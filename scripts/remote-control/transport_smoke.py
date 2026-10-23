"""First-task transport smoke policy, not a six-task solver or model benchmark.

Visits visible mint targets using decoded RGB pixels only. Supports Level I's
Move task; Level II additionally requires reading the requested color. Run verify-webrtc-browser.mjs with ARENA_TEST_EXPECT_PASSES=1, which
stops the challenge after this task and exports the applied-input diagnostics.
"""
import numpy


def decide(frame, controls):
    pixels = frame.to_ndarray(format="rgb24")[::4, ::4].astype(numpy.int16)
    r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    mask = (r >= 175) & (g >= 200) & (b >= 170) & (g-r > 20) & (g-b > 12)
    mask[:int(len(mask)*.25)] = False
    mask[int(len(mask)*.85):] = False
    height, width = mask.shape
    for y, x in zip(*numpy.where(mask)):
        if not mask[y, x]:
            continue
        queue, points = [(int(x), int(y))], []
        mask[y, x] = False
        while queue:
            px, py = queue.pop()
            points.append((px, py))
            for nx, ny in ((px-1, py), (px+1, py), (px, py-1), (px, py+1)):
                if 0 <= nx < width and 0 <= ny < height and mask[ny, nx]:
                    mask[ny, nx] = False
                    queue.append((nx, ny))
        if len(points) >= 12:
            # One target per received frame; visited targets change color.
            controls.moveTo(sum(p[0] for p in points)/len(points)/width,
                            sum(p[1] for p in points)/len(points)/height)
            return
