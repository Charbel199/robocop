import cv2
import numpy as np
import math


# Concat images and titles in a single window
def concat_images(images, titles, force_row_size=None, with_titles=True, width=None, height=None, fontScale=1):
    if width is None:
        width = images[0].shape[1]
    if height is None:
        height = images[0].shape[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (15, 35)
    fontColor = (0, 255, 0)
    thickness = 1
    lineType = 1
    for i, image in enumerate(images):
        # Make all images 3 channels
        images[i] = np.stack((image,) * 3, axis=-1) if len(image.shape) < 3 else image
        if images[i].shape != (height, width, 3):
            up_points = (width, height)
            images[i] = cv2.resize(images[i], up_points, interpolation=cv2.INTER_AREA)
        image = images[i]

        if with_titles:
            cv2.putText(image, titles[i],
                        position,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

    num_images = len(images)
    s = int(math.sqrt(num_images)) if force_row_size is None else force_row_size

    # Append empty images
    empty_images = s * math.ceil(num_images / s) - num_images
    [images.append(np.zeros_like(images[0])) for _ in range(empty_images)]

    def join_images(ims, horizontal=1):
        return np.concatenate(tuple(ims), axis=1 if horizontal else 0)

    split_arrays = np.array_split(images, s)
    vert_arrays = []
    for arr in split_arrays:
        vert_arrays.append(join_images(arr, horizontal=1))

    final_image = join_images(vert_arrays, horizontal=0)

    return final_image


