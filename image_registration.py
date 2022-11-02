import cv2
import numpy as np


def image_registration(img_1_clr, img_2_clr):
    img_1 = cv2.cvtColor(img_1_clr, cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(img_2_clr, cv2.COLOR_BGR2GRAY)
    height, width = img_2.shape

    # ORB detector
    orb_detector = cv2.ORB_create(15000)

    kp1, d1 = orb_detector.detectAndCompute(img_1, None)
    kp2, d2 = orb_detector.detectAndCompute(img_2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    matches.sort(key=lambda x: x.distance)

    matches = matches[: int(len(matches) * 0.7)]
    no_of_matches = len(matches)

    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    crop_1 = cv2.warpPerspective(img_1_clr, homography, (width, height))
    crop_2 = img_2_clr

    return crop_1, crop_2
