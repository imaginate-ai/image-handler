from image_handler.image_generator import ImageHandler

image_handler_instance = ImageHandler()


def get_image_handler():
    return image_handler_instance
