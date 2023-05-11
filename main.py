import cv2
import numpy as np
import DashcamPreprocessing as dcp

def main():
    img = cv2.imread('DashcamImages/color_image_0.jpg')
    print(img.shape)  # Print image shape
    cv2.imshow("original", img)

    croppedImgs = dcp.separateDashcamImg(img)

    for i in range(4):
        # Save the cropped image
        cv2.imwrite(f"Cropped Image-{i}.jpg", croppedImgs[i])

if __name__ == '__main__':
    main()