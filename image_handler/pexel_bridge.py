import os
from http import HTTPStatus

import requests
from image_handler.db import save_image
from image_handler.errors import InsufficientImagesError, TooManyRequestsError
from image_handler.util import get_image_from_url
from image_handler_client.schemas.image_info import ImageInfo, ImageStatus
from PIL import Image

PEXELS_BASE_URL = "https://api.pexels.com/v1"


def fetch_image(theme: str, num_images: int) -> dict | None:
    if num_images <= 0:
        return None

    params = {
        "query": theme,
        "per_page": num_images,
    }

    response = requests.get(
        f"{PEXELS_BASE_URL}/search",
        params=params,
        headers={
            "Authorization": os.getenv("PEXELS_TOKEN"),
            "user-agent": "Mozilla/5.0 Chrome/68.0.3440.106 Safari/537.36",
        },
        timeout=5,
    )

    if response.status_code == HTTPStatus.OK:
        return response.json()
    elif response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        raise TooManyRequestsError()
    elif response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR:
        raise ValueError(f"Internal server error: {response.text}")
    else:
        raise ValueError(f"Failed to fetch images: {response.text}")


def resize_image(image: Image) -> Image:
    """
    Resize an image to 512x512 pixels without distorting its aspect ratio.
    Crops the image to a centered square before resizing.

    :param image: Path to the input image file.
    :return: Resized image object.
    """
    min_side = min(image.size)

    left = (image.width - min_side) / 2
    top = (image.height - min_side) / 2
    right = (image.width + min_side) / 2
    bottom = (image.height + min_side) / 2

    img_cropped = image.crop((left, top, right, bottom))
    resized_img = img_cropped.resize((512, 512))

    return resized_img


def save_pexel_images(info: ImageInfo, num_images: int, skip: int = 0):
    response_data = fetch_image(info.theme, num_images + skip)
    if response_data is None:
        return

    if num_images > response_data["total_results"]:
        raise InsufficientImagesError(num_images, response_data["total_results"])

    photos_data = response_data["photos"]
    photo_urls = [photo["src"]["original"] for photo in photos_data]

    i = 0
    for url in photo_urls:
        if i < skip:
            i += 1
            continue
        image = get_image_from_url(url)
        new_image = resize_image(image)

        filename = f"{info.theme}_{int(info.filename.split("_")[-1] ) +i}"
        info_instance = ImageInfo(
            filename=filename,
            date=info.date,
            theme=info.theme,
            real=info.real,
            status=ImageStatus.UNVERIFIED.value,
        )
        save_image(new_image, info_instance)
        i += 1
