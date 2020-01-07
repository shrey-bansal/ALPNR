from skimage.io import imread
from skimage.util import img_as_ubyte


def binary_image():
    car_image = imread("cars.jpg", as_gray=True)
    # it would be a 2 dimensional array

    # the next line is not compulsory however, a grey scale pixel
    # in skimage ranges between 0 & 1. multiplying it with 255
    # will make it range between 0 & 255 (something we can relate better with
    gray_car_image = img_as_ubyte(car_image)
    threshold_value = 200
    binary_car_image = gray_car_image > threshold_value
    return binary_car_image


def grey_image():
    car_image = imread("cars.jpg", as_gray=True)
    gray_car_image = img_as_ubyte(car_image)
    return gray_car_image
