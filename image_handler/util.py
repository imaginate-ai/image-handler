import io
import os
import secrets
from datetime import datetime

import requests
from image_handler.constants import GREEN, MAX_IMAGES, RED, RESET, WHITE
from image_handler_client.schemas.image_info import ImageInfo, ImageStatus
from PIL import Image


def get_image_from_url(url: str) -> Image:
    headers = {
        "Authorization": os.getenv("PEXELS_TOKEN"),
        "User-Agent": "Mozilla/5.0 Chrome/58.0.3029.110 Safari/537.3",
    }
    response = requests.get(url, stream=True, headers=headers, timeout=5)

    return Image.open(io.BytesIO(response.content))


def print_date(date: int, data: list) -> bool:
    date_width = 15
    col_width = 10

    all_verified = True

    count = count_images(data)

    # create list of each image, colouring them based on their status
    real_ai_list = []
    for image in data:
        image_info = ImageInfo(**image)
        if image_info.status == ImageStatus.VERIFIED.value:
            colour = GREEN
        elif image_info.status == ImageStatus.REJECTED.value:
            colour = RED
            all_verified = False
        else:
            colour = WHITE
            all_verified = False

        real_ai_list.append(f"{colour}{'real' if image_info.real else 'ai':<{col_width}}{RESET}")

    # If the response is shorter than MAX_IMAGES, fill the remaining slots with blank space
    if len(real_ai_list) < MAX_IMAGES:
        real_ai_list.extend([f"{'':<{col_width}}"] * (MAX_IMAGES - len(real_ai_list)))
    real_ai_string = "".join(real_ai_list)

    date_str = f"{convert_to_readable_date(date)}:"
    date_str += " " * (date_width - len(date_str))

    # Determine if the day is a success
    if all_verified and len(data) == MAX_IMAGES and count["real"] > 0 and count["ai"] > 0:
        status = "SUCCESS"
        colour = GREEN
    elif (
        any(image["status"] == ImageStatus.REJECTED.value for image in data)
        or len(data) != MAX_IMAGES
        or count["real"] == 0
        or count["ai"] == 0
    ):
        status = "FAILURE"
        colour = RED
    else:
        status = "UNVERIFIED"
        colour = WHITE

    # Print the final formatted output with all images and the status
    print(f"{colour}{date_str}{RESET}{real_ai_string}{colour}{status}{RESET}")

    if status == "SUCCESS":
        return True
    if status == "UNVERIFIED" and count["real"] > 0 and count["ai"] > 0 and len(data) == MAX_IMAGES:
        return True
    return False


def count_images(response):
    real_count = 0
    ai_count = 0
    for image in response:
        if image["real"]:
            real_count += 1
        else:
            ai_count += 1

    return {"real": real_count, "ai": ai_count}


def convert_to_readable_date(unix_timestamp):
    return datetime.fromtimestamp(unix_timestamp).strftime("%Y-%m-%d")


def calculate_images_to_generate(real_count, ai_count):
    remaining_slots = MAX_IMAGES - (real_count + ai_count)

    # Ensure there's at least one of each type
    real_to_generate = 1 if real_count == 0 else 0
    ai_to_generate = 1 if ai_count == 0 else 0

    # Adjust remaining slots after ensuring at least one of each
    remaining_slots -= real_to_generate + ai_to_generate

    # Distribute remaining slots randomly if available
    if remaining_slots > 0:
        rand = secrets.randbelow(remaining_slots + 1)
        real_to_generate += rand
        ai_to_generate += remaining_slots - rand
    elif remaining_slots < 0:
        raise ValueError("No remaining slots but one type is missing")

    return {
        "real": real_to_generate,
        "ai": ai_to_generate,
    }
