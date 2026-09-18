"""Streaming adapter for the shared HMR2 crop pipeline."""

from lib.util.hmr2_preprocess import get_bbx_xys_from_xyxy, prepare_crops


def get_batch_from_frames(frames, bbx_xys, img_dst_size=256):
    return prepare_crops(frames, bbx_xys, size=img_dst_size)
