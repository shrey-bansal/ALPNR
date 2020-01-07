from skimage import measure
from skimage.transform import resize
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from Image_Preprocess import binary_image, grey_image

binary_car_image = binary_image()
gray_car_image = grey_image()

# this gets all the connected regions and groups them together
label_image = measure.label(binary_car_image)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_car_image, cmap="gray")
# regionprops creates a list of properties of all the labelled regions
license_plate = 0
for region in regionprops(label_image):
    if region.area < 100:
        # if the region is so small then it's likely not a license plate
        continue
    minr, minc, maxr, maxc = region.bbox
    height = binary_car_image.shape[0]
    width = binary_car_image.shape[1]
    if (maxc - minc) / (maxr - minr) < 1.5:
        continue
    if (maxc - minc > 0.5 * width) or (maxc - minc < 0.1 * width):
        continue
    if (maxr - minr > 0.15 * height) or (maxc - minc < 0.1 * height):
        continue
    if maxr < height / 2:
        continue
    # the bounding box coordinates
    minRow, minCol, maxRow, maxCol = region.bbox
    rectBorder = patches.Rectangle((minCol, minRow), maxCol - minCol, maxRow - minRow, edgecolor="red", linewidth=2,
                                   fill=False)
    ax1.add_patch(rectBorder)
    # let's draw a red rectangle over those regions
    license_plate = gray_car_image[minr:maxr, minc:maxc]
    license_plate = license_plate < 150
    ax2.imshow(license_plate, cmap="gray")
    segregation_image = measure.label(license_plate)
    for region1 in regionprops(segregation_image):
        minRow, minCol, maxRow, maxCol = region1.bbox
        height = segregation_image.shape[0]
        width = segregation_image.shape[1]
        if maxCol - minCol > 0.1 * width:
            continue
        if region1.area < height * width * 0.005:
            continue
        image_resized = resize(segregation_image[maxRow - minRow, maxCol - minCol], (20, 20), anti_aliasing=True)
        print(image_resized.shape)
        rectBorder = patches.Rectangle((minCol, minRow), maxCol - minCol, maxRow - minRow, edgecolor="red", linewidth=2,
                                       fill=False)
        ax2.add_patch(rectBorder)
plt.show()
