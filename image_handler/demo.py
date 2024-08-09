from image_generator import ImageHandler
from image_handler_client.schemas.image_info import ImageInfo

image_handler = ImageHandler()

default_kwargs = {
    "width": 512,
    "height": 512,
    "num_inference_steps": 250,
    "guidance_scale": 7,
    "generator": None,
    "output_type": "pil",
    "enhance_prompt": "yes",
}

def generate(theme):
    prompt_info = image_handler.create_prompt(theme)
    print(f"prompt: {prompt_info['prompt']}")
    info = ImageInfo(
        filename="grilled_cheese_1.jpg", date='0', theme="grilled cheese", real=False
    )
    kwargs = default_kwargs.copy()
    kwargs["prompt"] = prompt_info["prompt"]
    kwargs["negative_prompt"] = prompt_info["negative_prompt"]
    image_handler.enqueue_prompt_to_image(
        kwargs=kwargs, info=info
    )

while True:
    theme = input("theme: ")
    if theme == "exit":
        break
    generate(theme=theme)

image_handler.stop_processing()