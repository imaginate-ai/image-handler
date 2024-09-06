import io
import os
import queue
import threading
from enum import Enum
from typing import Union

from PIL import Image, ImageOps
import requests
import torch
from aws_lambda_powertools import Logger
from image_handler_client.schemas.image_info import ImageInfo, ImageStatus

from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
    DiffusionPipeline,
)

from errors import InsufficientImagesError
from dotenv import load_dotenv

load_dotenv()
logger = Logger()
PEXELS_BASE_URL = "https://api.pexels.com/v1"
URL = "http://127.0.0.1:5000/image/create"


class DataType(Enum):
    IMAGE = "image"
    PROMPT = "prompt"


class ImageHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized for {self.device}")

        # https://huggingface.co/timbrooks/instruct-pix2pix
        image_model_id = "timbrooks/instruct-pix2pix"
        self.image_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            image_model_id, torch_dtype=torch.float16
        ).to(self.device)
        self.image_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.image_pipe.scheduler.config
        )

        # https://huggingface.co/runwayml/stable-diffusion-v1-5
        text_model_id = "runwayml/stable-diffusion-v1-5"
        logger.info(f"Using model id: {text_model_id}")
        self.text_pipe = DiffusionPipeline.from_pretrained(
            text_model_id, torch_dtype=torch.float16, safety_checker=None
        ).to(self.device)
        self.text_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.image_pipe.scheduler.config
        )

        # Initialize the asynchronous queue and thread
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()

    def get_image_from_pexel(self, info: ImageInfo, theme: str, number: int):
        response = requests.get(
            f"{PEXELS_BASE_URL}/search",
            params={"query": theme, "per_page": number},
            headers={"Authorization": os.getenv("PEXELS_TOKEN")},
        )
        response_data = response.json()

        if number > response_data["total_results"]:
            raise InsufficientImagesError(number, response_data["total_results"])

        photos_data = response_data["photos"]
        photo_urls = [photo["src"]["original"] for photo in photos_data]

        for url in photo_urls:
            save_image(url, info)

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
        logger.info(f"\nEnqueued: {prompt}")

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
        logger.info(f"\nEnqueued: {prompt}")

    # Continuously process the queue
    def _process_queue(self):
        while True:
            task = self.queue.get()
            if task is None:
                break
            else:
                logger.info(f"\nDequeued: {task.get('kwargs', {}).get('prompt')}")
                logger.info(f"Task: {task}")

                if task.get("type") == DataType.IMAGE:
                    if "image" not in task["kwargs"]:
                        raise ValueError("Image is None")
                elif task.get("type") == DataType.PROMPT:
                    if "prompt" not in task["kwargs"]:
                        raise ValueError("Prompt is None")
                else:
                    logger.info("here")
                    raise ValueError("Invalid task type")
                self.process(task)
                self.queue.task_done()

    # Process the task
    def process(self, item):
        image = None
        if item["type"] == DataType.IMAGE:
            image = self.image_pipe(**item["kwargs"]).images[0]
        elif item["type"] == DataType.PROMPT:
            image = self.text_pipe(**item["kwargs"]).images[0]

        save_image(image, item["info"])
        remaining_tasks = self.queue.qsize() - 1
        logger.info(
            f"Saved image: {item['info'].filename}, {remaining_tasks} tasks remaining"
        )

    # Stop processing the queue, wait for all tasks to finish
    def stop_processing(self):
        size = self.queue.qsize() + 1 if self.queue.qsize() > 0 else 0
        logger.info(f"\nWaiting for {size} tasks to finish...")
        self.queue.put(None)
        self.thread.join()


# save image to database
def save_image(image: Union[Image.Image, str], info: ImageInfo):
    if image is Image.Image:
        image_type = "jpeg"
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image_type)
        image_bytes.seek(0)

        files = {"file": (info.filename, image_bytes, f"image/{image_type}")}

        response = requests.post(
            URL,
            {
                "real": False,
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
                "real": True,
                "date": info.date,
                "theme": info.theme,
                "status": ImageStatus.UNVERIFIED.value,
            },
            files=None,
        )

    status = response.status_code
    if status == 200:
        image_id = response.json()["url"].split("/")[-1]
        logger.info(f"Image saved with id: {image_id}")
    else:
        logger.error("Failed to save image")
        logger.error(response.text)
