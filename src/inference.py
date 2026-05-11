import os
from model import load_pretrained_model, load_trained_model


def run_pretrained_inference(config):
    """
    Step 3: Run inference with pretrained model.
    """

    model = load_pretrained_model(config.MODEL_NAME)

    image_dir = os.path.join(config.DATA_DIR, "test", "images")

    if not os.path.exists(image_dir):
        image_dir = os.path.join(config.DATA_DIR, "valid", "images")
        print("No test/images folder found. Using valid/images instead.")

    if not os.path.exists(image_dir):
        raise FileNotFoundError("No test/images or valid/images folder found.")

    results = model.predict(
        source=image_dir,
        imgsz=config.IMAGE_SIZE,
        conf=config.CONF_THRESHOLD,
        save=True,
        project=config.OUT_DIR,
        name="pretrained_inference",
        exist_ok=True
    )

    print("Pretrained inference completed.")
    print(f"Saved results to: {os.path.join(config.OUT_DIR, 'pretrained_inference')}")

    return results


def run_post_training_inference(config):
    """
    Step 5: Run inference with fine-tuned model on the same split as baseline.
    """

    model = load_trained_model(config)
    image_dir = os.path.join(config.DATA_DIR, "test", "images")

    if not os.path.exists(image_dir):
        image_dir = os.path.join(config.DATA_DIR, "valid", "images")
        print("No test/images folder found. Using valid/images instead.")

    if not os.path.exists(image_dir):
        raise FileNotFoundError("No test/images or valid/images folder found.")

    results = model.predict(
        source=image_dir,
        imgsz=config.IMAGE_SIZE,
        conf=config.CONF_THRESHOLD,
        save=True,
        project=config.OUT_DIR,
        name="post_training_inference",
        exist_ok=True,
    )

    print("Post-training inference completed.")
    print(f"Saved results to: {os.path.join(config.OUT_DIR, 'post_training_inference')}")

    return results