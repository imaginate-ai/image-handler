from image_handler.image_generator import ImageHandler

image_handler = None


def get_image_handler():
    global image_handler
    if image_handler is None:
        image_handler = ImageHandler()
    return image_handler
