import cv2
from tqdm import tqdm

from helpers import diff_image
from image_registration import image_registration

if __name__ == "__main__":
    # fake/origin
    for i in tqdm(range(1, 10)):
        path_1 = f"data/marks/fake_{i}.png"
        path_2 = f"data/marks/orig_{i}.png"
        img_1 = cv2.imread(path_1)
        img_2 = cv2.imread(path_2)

        crop_1, crop_2 = image_registration(img_1, img_2)

        diff = diff_image(crop_1, crop_2)

        cv2.imwrite(f"data/diff/diff_{i}.png", diff)

    # origin/origin
    for i in tqdm(range(5, 10)):
        path_1 = f"data/marks/orig_{i}.png"
        path_2 = f"data/marks/orig_{i}_2.png"
        img_1 = cv2.imread(path_1)
        img_2 = cv2.imread(path_2)

        crop_1, crop_2 = image_registration(img_1, img_2)

        diff = diff_image(crop_1, crop_2)

        cv2.imwrite(f"data/diff/diff_orig_{i}.png", diff)
