import cv2
from serl_robot_infra.so101_env.camera.usb_capture import USBCapture
import numpy as np


def preprocess(img: np.ndarray, blur_ksize: int = 15) -> np.ndarray:
    if blur_ksize > 1:
        return cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    return img


def crop_to_aspect_ratio(image, target_ratio):
    """
    Crops an image from the center to match a given target aspect ratio (width / height).
    """
    height, width = image.shape[:2]
    current_ratio = width / height

    if current_ratio > target_ratio:
        # Image is wider than target; crop the width
        new_width = int(height * target_ratio)
        offset = (width - new_width) // 2
        cropped_image = image[:, offset : offset + new_width]
    else:
        # Image is taller than target; crop the height
        new_height = int(width / target_ratio)
        offset = (height - new_height) // 2
        cropped_image = image[offset : offset + new_height, :]

    return cropped_image

if __name__ == "__main__":
    u = USBCapture("cam_wrist")

    ret, original_img = u.read()
    if not ret or original_img is None:
        raise Exception("Not read")

    h, w, c = original_img.shape
    dh, dw = 480, 640
    # assert dh <= h and dw <= w
    cropped_img = crop_to_aspect_ratio(original_img, dw / dh)
    img = cv2.resize(cropped_img, (640, 480))
    img = preprocess(img)
    cv2.imshow("blurred", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
