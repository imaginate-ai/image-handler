from image_handler.image_generator import ImageHandler

image_handler = None
sync = True


def get_image_handler():
    global image_handler
    if image_handler is None:
        image_handler = ImageHandler(sync=sync)
    return image_handler
