import secrets

from image_handler_client.schemas.image_info import ImageInfo, ImageStatus

from image_generator import ImageHandler, get_date, get_images_from_date

image_handler = ImageHandler()
MAX_IMAGES = 5
GUIDANCE_SCALE = 7
NUM_INFERENCE_STEPS = 2

# ANSI escape codes for colors
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def normal_run(theme: str):
    # get the date to complete
    date = get_date().get("date")
    if not date:
        date = 0
    date += 1

    num_pexel_images = secrets.randbelow(4) + 1
    num_ai_images = 5 - num_pexel_images

    add_images(theme, date, num_ai_images, num_pexel_images, 0)


def add_images(theme: str, date: int, num_ai_images: int, num_pexel_images: int, starting_number: int = 0):
    print(f"Saving to date: {date}")
    print(f"Getting {num_pexel_images} Pexel images")
    print(f"Generating {num_ai_images} AI images")

    image_handler.get_image_from_pexel(
        info=ImageInfo(
            filename=f"{theme}_{starting_number}",
            date=date,
            theme=theme,
            real=True,
            status=ImageStatus.UNVERIFIED.value,
        ),
        theme=theme,
        num_images=num_pexel_images,
    )

    for i in range(num_ai_images):
        prompt_dict = image_handler.create_prompt(theme=theme)
        print(f"Using prompt: {prompt_dict['prompt']}")
        image_handler.enqueue_prompt_to_image(
            info=ImageInfo(
                filename=f"{theme}_{i+num_pexel_images+starting_number}",
                date=date,
                theme=theme,
                real=False,
                status=ImageStatus.UNVERIFIED.value,
            ),
            kwargs={
                "prompt": prompt_dict["prompt"],
                "negative_prompt": prompt_dict["negative_prompt"],
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
                "width": 512,
                "height": 512,
            },
        )

    image_handler.stop_processing()


def count_images(response):
    real_count = 0
    ai_count = 0
    for image in response:
        if image["real"]:
            real_count += 1
        else:
            ai_count += 1
    return {"real": real_count, "ai": ai_count}


def print_date(date: int):
    response = get_images_from_date(date)

    colour = RED
    if response and len(response) == MAX_IMAGES:
        count = count_images(response)
        if count["real"] > 0 and count["ai"] > 0:
            colour = GREEN

    col_width = 20
    print("".join(f"{colour}{image['real']:<{col_width}}{RESET}" for image in response))

    print(f"response: {response}")


def check_latest():
    date = get_date().get("date")
    response = get_images_from_date(date)

    # early exit, we can assume there are no problems
    if len(response) == MAX_IMAGES:
        print("Up to date")
        return

    count = count_images(response)
    real_count = count["real"]
    ai_count = count["ai"]

    num_ai_images = secrets.randbelow(5 - real_count) + 1 - ai_count
    num_pexel_images = 5 - num_ai_images - real_count

    add_images(response[0]["theme"], date, num_ai_images, num_pexel_images)


if __name__ == "__main__":
    # check_latest()
    # user_theme = input("Enter theme: ")
    # normal_run(user_theme)
    print_date(1725768005)


# change the add images function to take date, num of both images
# that way we can have two intermediate functions, check latest and normal, run one after another
