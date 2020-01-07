from skimage.transform import resize
from skimage.measure import regionprops
from skimage import measure
from skimage.util import img_as_ubyte


def Plate_Letter_List(License_Plate_Gray):
    Letter_List = []
    License_Plate = License_Plate_Gray < 150
    segregation_image = measure.label(License_Plate)
    for region in regionprops(segregation_image):
        minRow, minCol, maxRow, maxCol = region.bbox
        height = segregation_image.shape[0]
        width = segregation_image.shape[1]
        if maxCol - minCol > 0.1 * width:
            continue
        if region.area < height * width * 0.005:
            continue
        image_resized = resize(License_Plate_Gray[minRow:maxRow, minCol:maxCol], (20, 20), anti_aliasing=True)
        image_resized = img_as_ubyte(image_resized)
        image_resized = image_resized <140
        Letter_List.append(image_resized)
    return Letter_List
