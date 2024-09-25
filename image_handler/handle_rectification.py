from image_handler.constants import GUIDANCE_SCALE, NUM_INFERENCE_STEPS
from image_handler.db import delete_rejected_images, save_image
from image_handler.image_handler_instance import get_image_handler
from image_handler.ollama_bridge import create_prompt
from image_handler.pexel_bridge import fetch_image, resize_image, save_pexel_images
from image_handler.util import calculate_images_to_generate, convert_to_readable_date, count_images, get_image_from_url
from image_handler_client.schemas.image_info import ImageInfo, ImageStatus


def handle_missing_images(date, data):
    count = count_images(data)
    real_count = count["real"]
    ai_count = count["ai"]
    generate = calculate_images_to_generate(real_count, ai_count)
    print(
        f"Generating {generate['ai']} AI images and "
        f"{generate['real']} real images for date: {convert_to_readable_date(date)}"
    )
    generate["real"] += generate["ai"]
    generate["ai"] = 0

    real_counter = 5
    for i in range(generate["real"]):
        response_data = fetch_image(data[0]["theme"], 80)
        photos_data = response_data["photos"]
        photo_urls = [photo["src"]["original"] for photo in photos_data]

        answer = "n"
        while answer != "y":
            image = resize_image(get_image_from_url(photo_urls[real_counter]))
            image.show()
            real_counter += 1

            answer = input(f'accept image for theme [{data[0]["theme"]}]? y/n: ')
            if answer == "y":
                info = ImageInfo(
                    filename="filename",
                    date=date,
                    theme=data[0]["theme"],
                    real=True,
                    status=ImageStatus.VERIFIED.value,
                )
                save_image(image, info)
                break

    for i in range(generate["ai"]):
        image_handler_instance = get_image_handler()

        prompt_dict = create_prompt(theme=data[0]["theme"])
        print(f"Using prompt: {prompt_dict['prompt']}")
        image_handler_instance.enqueue_prompt_to_image(
            info=ImageInfo(
                filename="filename",
                date=date,
                theme=data[0]["theme"],
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

        image_handler_instance.stop_processing()


def handle_rejected_images(date: int, data: list):
    real_counter = 5

    if not any(image["status"] == ImageStatus.REJECTED.value for image in data):
        return

    # replace rejected images
    delete_rejected_images(date)

    for image_data in data:
        image_info = ImageInfo(**image_data)

        # handle real images
        if image_info.status == ImageStatus.REJECTED.value:
            if image_info.real:
                response_data = fetch_image(image_info.theme, 80)
                photos_data = response_data["photos"]
                photo_urls = [photo["src"]["original"] for photo in photos_data]

                answer = "n"
                while answer != "y":
                    image = resize_image(get_image_from_url(photo_urls[real_counter]))
                    image.show()
                    real_counter += 1

                    answer = input("accept? y/n: ")
                    if answer == "y":
                        info = ImageInfo(
                            filename=image_info.filename,
                            date=date,
                            theme=image_info.theme,
                            real=True,
                            status=ImageStatus.VERIFIED.value,
                        )
                        save_image(image, info)
                        break

            # handle AI images
            else:
                image_handler_instance = get_image_handler()

                prompt_dict = create_prompt(theme=image_info.theme)
                print(f"Using prompt: {prompt_dict['prompt']}")
                image_handler_instance.enqueue_prompt_to_image(
                    info=ImageInfo(
                        filename=f"{image_info.theme}_{image_info.filename.split('_')[-1]}",
                        date=date,
                        theme=image_info.theme,
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

                image_handler_instance.stop_processing()


def add_images(theme: str, date: int, num_ai_images: int, num_pexel_images: int, starting_number: int = 0):
    image_handler_instance = get_image_handler()

    save_pexel_images(
        info=ImageInfo(
            filename=f"{theme}_{starting_number}",
            date=date,
            theme=theme,
            real=True,
            status=ImageStatus.UNVERIFIED.value,
        ),
        num_images=num_pexel_images,
        skip=starting_number,
    )

    for i in range(num_ai_images):
        prompt_dict = create_prompt(theme=theme)
        print(f"Using prompt: {prompt_dict['prompt']}")

        image_handler_instance.enqueue_prompt_to_image(
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
