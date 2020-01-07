from skimage import measure
from skimage.transform import resize
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from Image_Preprocess import binary_image, grey_image
from License_Plate_Detection import License_Plate_Image
from Digit_Segregation import Plate_Letter_List

binary_car_image = binary_image()
gray_car_image = grey_image()

License_Plate = License_Plate_Image(binary_car_image, gray_car_image)
x = License_Plate


Plate_Letters = Plate_Letter_List(License_Plate)
for i in range(len(Plate_Letters)):
    plt.imshow(Plate_Letters[i], cmap="gray")
    plt.show()
