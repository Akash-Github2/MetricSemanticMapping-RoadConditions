import cv2

#15360 width x 2160 height (img shape is height x width x rgb)
def separateDashcamImg(img): #is 4 images combined - so separate into four different images and return
    return [img[:, 0:3840], img[:, 3840:7680], img[:, 7680:11520], img[:, 11520:]]