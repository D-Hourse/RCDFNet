"""No-op plot helpers to keep training/inference import-compatible."""


def _noop(*args, **kwargs):
    return None


draw_feature_fuser_map = _noop
draw_feature_pts_map = _noop
draw_feature_img_LSS_map = _noop
draw_feature_img_OFT_map = _noop
draw_feature_img_fuser_map = _noop
draw_feature_map = _noop
save_image_tensor2cv2 = _noop
