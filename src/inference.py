import os
from model import Model


def run_pretrained_inference(config):
    """
    Step 3: Run inference with pretrained YOLO11n.
    """

    model = Model(config.MODEL_NAME).get_model()

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