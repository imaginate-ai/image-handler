from image_handler.constants import SECONDS_PER_DAY, START_DATE
from image_handler.db import get_date, get_grouped_images_by_date
from image_handler.handle_rectification import handle_missing_images, handle_rejected_images
from image_handler.util import print_date


def health_check(fix=False):
    end_date = get_date().get("date")
    if not end_date:
        print("No images in database")
        return

    data = get_grouped_images_by_date()

    print(f"There are {((end_date - START_DATE) // SECONDS_PER_DAY)+1} days to check, from {START_DATE} to {end_date}")

    current_date = START_DATE
    while current_date <= end_date:
        valid = print_date(current_date, data[current_date])

        if fix and not valid:
            handle_rejected_images(current_date, data[current_date])
            handle_missing_images(current_date, data[current_date])

        # Increment date by one day (86400 seconds)
        current_date += SECONDS_PER_DAY


if __name__ == "__main__":
    health_check(fix=True)
