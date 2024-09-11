import http.client
import io
import json
import os
import queue
import sys
import threading
from enum import Enum
from typing import Union

import httpx
import ollama
import requests
import torch
from aws_lambda_powertools import Logger
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
)
from dotenv import load_dotenv
from image_handler_client.schemas.image_info import ImageInfo, ImageStatus
from PIL import Image, ImageOps

from errors import InsufficientImagesError

load_dotenv()
logger = Logger()
PEXELS_BASE_URL = "https://api.pexels.com/v1"
URL = "http://127.0.0.1:5000/image/create"
DATE_URL = "http://127.0.0.1:5000/date/latest"
DATE_URL = "http://127.0.0.1:5000/date"


class DataType(Enum):
    IMAGE = "image"
    PROMPT = "prompt"


class ImageHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized for {self.device}")

        # Initialize the asynchronous queue and thread
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()

    # create prompt from theme input
    def create_prompt(self, theme: str) -> dict:
        try:
            response = ollama.chat(
                model="llama3",
                keep_alive=0,
                messages=[
                    {
                        "role": "user",
                        "content": f"generate a prompt for an ai image model to create a real photograph of {theme}. "
                        f"Only output the prompt, nothing else, as the response will be fed "
                        f"directly to the image generator. Make sure the prompt is under 75 words.",
                    },
                ],
            )
        except httpx.ConnectError:
            print("Error: 'ollama' service needs to be running. Please start it and try again.")
            sys.exit(1)

        prompt_info = {
            "prompt": response["message"]["content"],
            "negative_prompt": (
                "bad lighting, out of focus, blurred, poorly composed, low resolution, "
                "extra elements, hands, people, non-food items, cartoonish, animated, "
                "unrealistic, uncentered, over saturated zoomed in"
            ),
        }
        return prompt_info

    def resize_image(self, image):
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

    def get_image_from_pexel(self, info: ImageInfo, theme: str, num_images: int):
        response = requests.get(
            f"{PEXELS_BASE_URL}/search",
            params={
                "query": theme,
                "per_page": num_images,
                "orientation": "square",
            },
            headers={"Authorization": os.getenv("PEXELS_TOKEN")},
            timeout=5,
        )
        response_data = response.json()

        if num_images > response_data["total_results"]:
            response = requests.get(
                f"{PEXELS_BASE_URL}/search",
                params={
                    "query": theme,
                    "per_page": num_images - response_data["total_results"],
                },
                headers={"Authorization": os.getenv("PEXELS_TOKEN")},
                timeout=5,
            )
            new_response_data = response.json()

            response_data["photos"].extend(new_response_data["photos"])
            response_data["total_results"] += len(new_response_data["photos"])

            if num_images > response_data["total_results"]:
                raise InsufficientImagesError(num_images, response_data["total_results"])

        photos_data = response_data["photos"]
        photo_urls = [photo["src"]["original"] for photo in photos_data]

        i = 0
        for url in photo_urls:
            image = self.get_image_from_url(url)
            new_image = self.resize_image(image)

            filename = f"{theme}_{int(info.filename.split("_")[-1])+i}"
            info_instance = ImageInfo(
                filename=filename,
                date=info.date,
                theme=info.theme,
                real=info.real,
                status=ImageStatus.UNVERIFIED.value,
            )
            save_image(new_image, info_instance)
            i += 1

    # get image from url
    def get_image_from_url(self, url: str) -> Image.Image:
        image = Image.open(requests.get(url, stream=True, timeout=10).raw)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image

    # add image to image task to queue
    def enqueue_image_to_image(
        self,
        info: ImageInfo,
        image: Union[Image.Image, str],
        prompt="make photograph",
        num_inference_steps=None,
        image_guidance_scale=None,
    ):
        # pylint: disable=unused-argument

        if isinstance(image, str):
            image = self.get_image_from_url(image)

        if image is None:
            raise ValueError("Image is None")

        kwargs = {k: v for k, v in locals().items() if v and v is not self}
        task = {
            "type": DataType.IMAGE,
            "kwargs": kwargs,
            "info": info,
        }
        self.queue.put(task)
        logger.info(f"\nEnqueued: {info.filename}")

    # add text to image task to queue
    def enqueue_prompt_to_image(
        self,
        info: ImageInfo,
        prompt: str = None,
        negative_prompt=None,
        extra_prompt=None,
        num_inference_steps=None,
        guidance_scale=None,
        kwargs=None,
    ):
        # pylint: disable=unused-argument

        if not kwargs:
            if prompt is None:
                raise ValueError("Prompt is None")

            kwargs = {k: v for k, v in locals().items() if v and v is not self}
        task = {
            "type": DataType.PROMPT,
            "kwargs": kwargs,
            "info": info,
        }
        self.queue.put(task)
        logger.info(f"\nEnqueued: {info.filename}")

    # Continuously process the queue
    def _process_queue(self):
        while True:
            task = self.queue.get()
            if task is None:
                break
            else:
                logger.info(f"\nDequeued: {task.get('info').filename}")

                if task.get("type") == DataType.IMAGE:
                    if "image" not in task["kwargs"]:
                        raise ValueError("Image is None")
                elif task.get("type") == DataType.PROMPT:
                    if "prompt" not in task["kwargs"]:
                        raise ValueError("Prompt is None")
                else:
                    raise ValueError("Invalid task type")
                self.process(task)
                self.queue.task_done()

    # Process the task
    def process(self, item):
        image = None
        if item["type"] == DataType.IMAGE:
            image = self.process_image(item)
        elif item["type"] == DataType.PROMPT:
            image = self.process_prompt(item)

        save_image(image, item["info"])

    # Stop processing the queue, wait for all tasks to finish
    def stop_processing(self):
        size = self.queue.qsize() + 1 if self.queue.qsize() > 0 else 0
        logger.info(f"\nWaiting for {size} tasks to finish...")
        self.queue.put(None)
        self.thread.join()

    def process_image(self, item):
        # https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
        image_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
        image_pipe = DiffusionPipeline.from_pretrained(image_model_id, torch_dtype=torch.float16).to(self.device)
        image_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(image_pipe.scheduler.config)

        return image_pipe(**item["kwargs"]).images[0]

    def process_prompt(self, item):
        # https://huggingface.co/black-forest-labs/FLUX.1-schnell
        text_model_id = "stabilityai/sdxl-turbo"
        text_pipe = DiffusionPipeline.from_pretrained(text_model_id, torch_dtype=torch.float16, safety_checker=None).to(
            self.device
        )
        text_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(text_pipe.scheduler.config)

        return text_pipe(**item["kwargs"]).images[0]


# save image to database
def save_image(image: Union[Image.Image, str], info: ImageInfo):
    if isinstance(image, Image.Image):
        image_type = "jpeg"
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image_type)
        image_bytes.seek(0)

        files = {"file": (info.filename, image_bytes, f"image/{image_type}")}

        response = requests.post(
            URL,
            {
                "real": info.real,
                "date": info.date,
                "theme": info.theme,
                "status": ImageStatus.UNVERIFIED.value,
            },
            files=files,
            timeout=10,
        )
    else:
        response = requests.post(
            URL,
            {
                "url": image,
                "real": info.real,
                "date": info.date,
                "theme": info.theme,
                "status": ImageStatus.UNVERIFIED.value,
            },
            files=None,
            timeout=5,
        )

    status = response.status_code
    if status == http.client.OK:
        logger.info(f"Image [{info.filename}] successfully saved")
    else:
        logger.error(f"Failed to save image [{info.filename}]")
        logger.error(response.text)


def get_date():
    response = requests.get(
        f"{DATE_URL}/latest",
        timeout=10,
    )
    return json.loads(response.text)


def get_images_from_date(day: int):
    response = requests.get(f"{DATE_URL}/{day}/images", timeout=10)
    return json.loads(response.text)
