from image_handler_client.schemas.image_info import ImageInfo

from image_generator import ImageHandler

image_handler = ImageHandler()

default_kwargs = {
    "width": 512,
    "height": 512,
    "num_inference_steps": 25,
    "guidance_scale": 7,
    "generator": None,
    "output_type": "pil",
    "enhance_prompt": "yes",
}


def example():
    info = ImageInfo(filename="grilled_cheese_1.jpg", date=0, theme="grilled cheese", real=False)
    kwargs = default_kwargs.copy()
    kwargs["prompt"] = (
        "ultra realistic single grilled cheese sandwich, zoomed out ((melted cheese oozing from toasted bread)), "
        "crispy golden-brown crust, hyper detail, cinematic lighting, Canon EOS R3, nikon, f/1.4, ISO 200, 1/160s, "
        "8K, RAW, unedited, in-frame"
    )
    kwargs["negative_prompt"] = (
        "burnt, overcooked, soggy, poorly prepared, bad lighting, out of focus, blurred, poorly composed, "
        "low resolution, extra elements, hands, people, non-food items, cartoonish, animated, unrealistic, "
        "uncentered, over saturated zoomed in"
    )
    image_handler.enqueue_prompt_to_image(kwargs=kwargs, info=info)

    image_handler.stop_processing()


if __name__ == "__main__":
    CHOICE = str(input("Do you want to run 'pexels', 'example', 'live', or 'check db'? ")).lower()
    if CHOICE == "example":
        example()
    if CHOICE == "live":
        while True:
            PROMPT = str(input("Enter prompt (exit to quit): "))
            if PROMPT != "exit":
                image_handler.enqueue_prompt_to_image(
                    prompt=PROMPT,
                    info=ImageInfo(filename="live.jpg", date=0, theme="live", real=False),
                )
            else:
                break

        image_handler.stop_processing()
    if CHOICE == "check db":
        import requests

        response = requests.get("http://127.0.0.1:5000/image/read", timeout=10)
        print(response.text)
    if CHOICE == "pexels":
        theme = input("Enter theme: ")
        info = ImageInfo(filename="fake", date="0", theme=theme, real=False)
        image_handler.get_image_from_pexel(info, theme, 3)
