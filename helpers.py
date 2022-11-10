import cv2
import imutils as imutils
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm


def vanilla_crop(img, thresh, n=13):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv_img[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    equ = clahe.apply(s)

    equ = cv2.GaussianBlur(equ, (n, n), 0)

    im_b = cv2.threshold(equ, thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    im_b = cv2.erode(im_b, None, iterations=2)
    im_b = cv2.dilate(im_b, None, iterations=2)

    crop_img = img[np.ix_(im_b.any(1), im_b.any(0))]

    return crop_img


def crop_stamps(img, thresh=120):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    s = hsv_img[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    equ = clahe.apply(s)

    im_b = cv2.adaptiveThreshold(
        equ, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 0
    )

    try:
        all_contours = cv2.findContours(im_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        all_contours = imutils.grab_contours(all_contours)
        all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)

        perimeter = cv2.arcLength(all_contours[0], True)
        ROIdimensions = cv2.approxPolyDP(all_contours[0], 0.02 * perimeter, True)
        ROIdimensions = ROIdimensions.reshape(4, 2)

    except:
        print("vanilla")
        return vanilla_crop(img, thresh)

    rect = np.zeros((4, 2), dtype="float32")
    s = np.sum(ROIdimensions, axis=1)
    rect[0], rect[2] = ROIdimensions[np.argmin(s)], ROIdimensions[np.argmax(s)]

    diff = np.diff(ROIdimensions, axis=1)
    rect[1], rect[3] = ROIdimensions[np.argmin(diff)], ROIdimensions[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    width_a = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
    width_b = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    height_b = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    transformMatrix = cv2.getPerspectiveTransform(rect, dst)
    scan = cv2.warpPerspective(img, transformMatrix, (max_width + 10, max_height + 10))

    return scan


def findHomography(img1, img2, MIN_MATCH_COUNT = 10, MIN_DIST_THRESHOLD = 0.9, RANSAC_REPROJ_THRESHOLD = 5.0):

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 2
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=150)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < MIN_DIST_THRESHOLD * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
        return H


def resize_stamps(img_1, img_2, thresh=120, MIN_MATCH_COUNT=10, MIN_DIST_THRESHOLD=0.9):
    crop_1 = crop_stamps(img_1, thresh)
    crop_2 = crop_stamps(img_2, thresh)

    H = findHomography(crop_1, crop_2, MIN_MATCH_COUNT=MIN_MATCH_COUNT, MIN_DIST_THRESHOLD=MIN_DIST_THRESHOLD)
    crop_1 = cv2.warpPerspective(crop_1, H, (crop_2.shape[1], crop_2.shape[0]))

    return crop_1, crop_2


def diff_image(img_1, img_2, thresh=100):

    gray_a = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.bilateralFilter(gray_a, 50, 17, 17)

    gray_b = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.bilateralFilter(gray_b, 50, 17, 17)

    (score, diff) = compare_ssim(gray_a, gray_b, full=True)
    diff = (diff * 255).astype("uint8")

    diff = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)[1]

    return diff, score


if __name__ == "__main__":
    for i in tqdm(range(1, 8), position=0, leave=True):
        img_1 = cv2.imread(f"data/marks/orig_{i}.png", 1)
        print(img_1.shape)
        img_2 = cv2.imread(f"data/marks/fake_{i}.png", 1)

        crop_1, crop_2 = resize_stamps(img_1, img_2, thresh=120, MIN_MATCH_COUNT=10, MIN_DIST_THRESHOLD=0.9)

        cv2.imwrite(f"data/marks_v1/orig_{i}.png", crop_1)
        cv2.imwrite(f"data/marks_v1/fake_{i}.png", crop_2)
