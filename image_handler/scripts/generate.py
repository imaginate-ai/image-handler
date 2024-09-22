from image_handler.constants import SECONDS_PER_DAY, START_DATE
from image_handler.db import get_date, get_grouped_images_by_date
from image_handler.handle_rectification import add_images, handle_missing_images
from image_handler.image_handler_instance import get_image_handler
from image_handler.util import calculate_images_to_generate, print_date


def check_last_date():
    date = get_date().get("date")
    if not date:
        print("No images in database")
        return

    data = get_grouped_images_by_date()

    valid = print_date(date, data[date])
    if valid:
        return

    handle_missing_images(date, data[date])


def generate_images(theme: str):
    date = get_date().get("date")
    if not date:
        date = START_DATE
    else:
        date += SECONDS_PER_DAY

    generate = calculate_images_to_generate(0, 0)

    add_images(
        theme=theme,
        date=date,
        num_ai_images=generate["ai"],
        num_pexel_images=generate["real"],
        starting_number=0,
    )


if __name__ == "__main__":
    check_last_date()

    themes = [
        "cat",
        "dog",
        "flower",
        "car",
    ]

    for theme in themes:
        generate_images(theme)

    get_image_handler().stop_processing()
