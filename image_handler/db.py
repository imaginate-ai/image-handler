import http.client
import io
import json
from http import HTTPStatus
from typing import Union

import requests
from image_handler.constants import DATE_URL, IMAGE_URL
from image_handler_client.schemas.image_info import ImageInfo
from PIL import Image


def get_date():
    response = requests.get(
        f"{DATE_URL}/latest",
        timeout=10,
    )
    return json.loads(response.text)


def save_image(image: Union[Image.Image, str], info: ImageInfo):
    if isinstance(image, Image.Image):
        image_type = "jpeg"
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image_type)
        image_bytes.seek(0)

        files = {"file": (info.filename, image_bytes, f"image/{image_type}")}

        response = requests.post(
            f"{IMAGE_URL}/create",
            {
                "real": info.real,
                "date": info.date,
                "theme": info.theme,
                "status": info.status,
            },
            files=files,
            timeout=10,
        )
    else:
        response = requests.post(
            f"{IMAGE_URL}/create",
            {
                "url": image,
                "real": info.real,
                "date": info.date,
                "theme": info.theme,
                "status": info.status,
            },
            files=None,
            timeout=5,
        )

    status = response.status_code
    if status == http.client.OK:
        print(f"Image [{info.filename}] successfully saved")
    else:
        print(f"Failed to save image [{info.filename}]")
        print(response.text)


def get_grouped_images_by_date() -> dict:
    response = requests.get(f"{IMAGE_URL}/all-images", timeout=10)
    response = response.json()

    date_image_map = {}

    # Iterate through each image in the response
    for image in response:
        date = image["date"]

        # If the date is not already a key, create a new list for it
        if date not in date_image_map:
            date_image_map[date] = []

        # Append the image to the list associated with this date
        date_image_map[date].append(image)

    return date_image_map


def delete_rejected_images(date):
    """Send a DELETE request to delete the rejected image by date and filename."""
    try:
        response = requests.delete(
            f"{DATE_URL}/delete-rejected/{date}",
            timeout=10,
        )
        if response.status_code == HTTPStatus.OK:
            print(f"Successfully deleted rejected images for date: {date}")
        else:
            print(f"Failed to delete rejects for day: {date}. Status code: {response.status_code}")
            print(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error deleting image: {e}")
