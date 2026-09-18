"""RGB video access for preprocessing, backed by ImageIO and PyAV."""

from fractions import Fraction

import av
import imageio.v3 as iio
import numpy as np


def video_info(path):
    shape = iio.improps(str(path), plugin="pyav").shape
    return int(shape[0]), int(shape[2]), int(shape[1])


def read_rgb_video(path, scale=1.0):
    filters = [("scale", f"iw*{scale}:ih*{scale}")] if scale != 1.0 else None
    frames = list(iio.imiter(str(path), plugin="pyav", filter_sequence=filters))
    if not frames:
        raise ValueError(f"Video has no decodable frames: {path}")
    return np.stack(frames)


def write_rgb_video(frames, path, fps=30, crf=17):
    images = np.asarray(frames, dtype=np.uint8)
    if images.ndim != 4 or images.shape[-1] != 3 or not len(images):
        raise ValueError("Expected nonempty RGB frames [T, H, W, 3]")
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(str(fps)))
        stream.width, stream.height = images.shape[2], images.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf)}
        for image in images:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
