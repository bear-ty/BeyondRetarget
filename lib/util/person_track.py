"""Select and complete a single person's bounding-box track."""

import numpy as np
import torch
import torch.nn.functional as F


def complete_track(frame_ids, corners, frame_count):
    """Interpolate missing boxes, hold endpoint values, and smooth twice."""
    ids = torch.as_tensor(frame_ids, dtype=torch.long)
    boxes = torch.as_tensor(corners, dtype=torch.float32)
    if not len(ids) or boxes.shape != (len(ids), 4):
        raise ValueError("A track needs matching frame IDs and [N, 4] boxes")
    if ids[0] < 0 or ids[-1] >= frame_count or torch.any(ids[1:] <= ids[:-1]):
        raise ValueError("Track frame IDs must be increasing and within the video")
    dense = boxes.new_empty((frame_count, 4))
    dense[:ids[0]] = boxes[0]
    dense[ids[-1] + 1:] = boxes[-1]
    dense[ids] = boxes
    for index in range(len(ids) - 1):
        left, right = int(ids[index]), int(ids[index + 1])
        if right > left + 1:
            blend = torch.linspace(0, 1, right - left + 1)[1:-1, None]
            dense[left + 1:right] = blend * (boxes[index + 1] - boxes[index]) + boxes[index]
    channels = dense.transpose(0, 1).unsqueeze(1)
    kernel = boxes.new_full((1, 1, 5), 0.2)
    for _ in range(2):
        channels = F.conv1d(F.pad(channels, (2, 2), mode="replicate"), kernel)
    return channels.squeeze(1).transpose(0, 1)


def select_person_track(results, frame_count, width, height):
    """Choose the identity with greatest accumulated image-area coverage."""
    tracks = {}
    for frame_id, prediction in enumerate(results):
        if prediction.boxes.id is None:
            continue
        ids = prediction.boxes.id.int().cpu().tolist()
        corners = prediction.boxes.xyxy.cpu().numpy()
        for identity, box in zip(ids, corners):
            tracks.setdefault(identity, []).append((frame_id, box))
    if not tracks:
        raise RuntimeError("No person track detected in the video")

    def coverage(observations):
        boxes = np.stack([box for _, box in observations])
        sizes = boxes[:, 2:] - boxes[:, :2]
        return (sizes[:, 0] * sizes[:, 1] / width / height).sum()

    chosen = max(tracks.values(), key=coverage)
    return complete_track([frame for frame, _ in chosen], np.stack([box for _, box in chosen]), frame_count)
