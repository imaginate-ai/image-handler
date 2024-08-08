from image_generator import ImageHandler
from image_handler_client.schemas.image_info import ImageInfo
import ollama

image_handler = ImageHandler()

default_kwargs = {
    "width": 512,
    "height": 512,
    "num_inference_steps": 200,
    "guidance_scale": 7,
    "generator": None,
    "output_type": "pil",
    "enhance_prompt": "yes",
}

def generate(theme):
    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": f"generate a prompt for an ai image model to create a real photograph of {theme}. Only output the prompt, nothing else, as the response will be fed directly to the image generator. Make sure the prompt is under 75 characters.",
            },
        ],
    )
    prompt = response['message']['content']
    print(f"prompt: {prompt}")

    info = ImageInfo(
        filename="grilled_cheese_1.jpg", date='0', theme="grilled cheese", real=False
    )
    kwargs = default_kwargs.copy()
    kwargs["prompt"] = f"{prompt}"
    kwargs["negative_prompt"] = "bad lighting, out of focus, blurred, poorly composed, low resolution, extra elements, hands, people, non-food items, cartoonish, animated, unrealistic, uncentered, over saturated zoomed in"
    image_handler.enqueue_prompt_to_image(
        kwargs=kwargs, info=info
    )

while True:
    theme = input("theme: ")
    if theme == "exit":
        break
    generate(theme=theme)

image_handler.stop_processing()