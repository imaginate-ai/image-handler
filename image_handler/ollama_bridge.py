import httpx
import ollama
from image_handler.errors import OllamaError


def create_prompt(theme: str) -> dict:
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
        raise OllamaError()

    prompt_info = {
        "prompt": response["message"]["content"],
        "negative_prompt": (
            "bad lighting, out of focus, blurred, poorly composed, low resolution, "
            "extra elements, hands, people, non-food items, cartoonish, animated, "
            "unrealistic, uncentered, over saturated zoomed in"
        ),
    }
    return prompt_info
