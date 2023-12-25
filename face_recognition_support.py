def extract_1face_image(image, bounding_box):
    (x, y, w, h) = bounding_box

    image_cut = image[y:y+w, x:x+h, :]

    return image_cut