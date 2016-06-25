import cv2

class ImageProvider:

    @staticmethod
    def getImages():

        imgList = []
        img1 = cv2.imread("smiling-girl.jpg")
        print ("Shape of image1 - ", img1.shape)  # Color image is a 3 dimensional matrix
        imgList.append(img1)

        img2 = cv2.imread("smiling_boy.jpg")
        print ("Shape of image2 - ", img2.shape)
        imgList.append(img2)

        img3 = cv2.imread("crying_boy.jpg")
        print ("Shape of image3 - ", img3.shape)
        imgList.append(img3)

        return imgList