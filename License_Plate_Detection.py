from skimage import measure
from skimage.measure import regionprops


def License_Plate_Image(binary_car_image, gray_car_image):

    # this gets all the connected regions and groups them together

    label_image = measure.label(binary_car_image)

    # regionprops creates a list of properties of all the labelled regions

    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        height = binary_car_image.shape[0]
        width = binary_car_image.shape[1]
        if region.area < 100:
            continue
        if (maxc - minc) / (maxr - minr) < 1.5 or (maxc - minc)/(maxr - minr) > 8 :
            continue
        if (maxc - minc > 0.5 * width) or (maxc - minc < 0.1 * width):
            continue
        if (maxr - minr > 0.15 * height) or (maxc - minc < 0.1 * height):
            continue
        if maxr < height / 2:
            continue

        license_plate = gray_car_image[minr:maxr, minc:maxc]
        return license_plate
