import secrets

from image_handler_client.schemas.image_info import ImageInfo, ImageStatus

from image_generator import ImageHandler, get_date

image_handler = ImageHandler()
GUIDANCE_SCALE = None

NUM_INFERENCE_STEPS = 25


def add_images(theme: str):
    # get the date to complete
    date = get_date().get("date")
    if not date:
        date = 0

    # choose number of ai images to generate (1-4)
    num_ai_images = secrets.randbelow(4) + 1
    print(f"Generating {num_ai_images} AI images")

    for i in range(num_ai_images):
        prompt_dict = image_handler.create_prompt(theme=theme)
        image_handler.enqueue_prompt_to_image(
            info=ImageInfo(
                filename=f"{theme}_{i}",
                date=date,
                theme=theme,
                real=False,
                status=ImageStatus.UNVERIFIED.value,
            ),
            prompt=prompt_dict["prompt"],
            negative_prompt=prompt_dict["negative_prompt"],
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        )

    # do the rest as pexel images
    num_pexel_images = 5 - num_ai_images
    print(f"Getting {num_pexel_images} Pexel images")

    image_handler.get_image_from_pexel(
        info=ImageInfo(
            filename=f"{theme}_{num_ai_images}",
            date=date,
            theme=theme,
            real=True,
            status=ImageStatus.UNVERIFIED.value,
        ),
        theme=theme,
        num_images=num_pexel_images,
    )


if __name__ == "__main__":
    theme = input("Enter theme: ")
    add_images(theme)
