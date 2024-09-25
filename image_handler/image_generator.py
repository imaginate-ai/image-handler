import queue
import threading
from enum import Enum
from typing import Union

import torch
from aws_lambda_powertools import Logger
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
)
from dotenv import load_dotenv
from image_handler.db import save_image
from image_handler.util import get_image_from_url
from image_handler_client.schemas.image_info import ImageInfo
from PIL import Image

load_dotenv()
logger = Logger()


class DataType(Enum):
    IMAGE = "image"
    PROMPT = "prompt"


class ImageHandler:
    def __init__(self, sync=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized for {self.device}")

        self.sync = sync
        logger.info(f"Sync: {self.sync}")

        # Initialize the asynchronous queue and thread
        if sync:
            self.queue = queue.Queue()
            self.thread = threading.Thread(target=self._process_queue)
            self.thread.daemon = True
            self.thread.start()

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
            image = get_image_from_url(image)

        if image is None:
            raise ValueError("Image is None")

        kwargs = {k: v for k, v in locals().items() if v and v is not self}
        task = {
            "type": DataType.IMAGE,
            "kwargs": kwargs,
            "info": info,
        }

        print(f"\nEnqueued ai: {info.filename}")
        if not self.sync:
            self.queue.put(task)
        else:
            self.process(task)

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

        print(f"\nEnqueued ai: {info.filename}")
        if not self.sync:
            self.queue.put(task)
        else:
            self.process(task)

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
