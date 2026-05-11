import os
from ultralytics import YOLO


def load_pretrained_model(config):
    """Load a pretrained Ultralytics YOLO model by name or path."""
    path = os.path.join(config.MODEL_DIR, 'pre-trained', str(config.MODEL_NAME))
    return YOLO(path)


def load_trained_model(config):
    """Load the trained model checkpoint produced by YOLO training."""
    trained_model_path = os.path.join(config.MODEL_DIR, "training", "weights", "best.pt")
    if not os.path.exists(trained_model_path):
        raise FileNotFoundError(
            f"No trained model checkpoint found at '{trained_model_path}'."
        )
    return YOLO(trained_model_path)


def train_model(config, data_yaml: str):
    """Train the configured pretrained model on the given dataset."""
    model = load_pretrained_model(config)

    results = model.train(
        data=data_yaml,
        epochs=config.TRAIN_EPOCHS,
        imgsz=config.IMAGE_SIZE,
        batch=config.BATCH_SIZE,
        device=config.TRAIN_DEVICE,
        project=config.MODEL_DIR,
        name="training",
        exist_ok=True,
    )

    print("Training completed.")
    print(f"Saved checkpoints to: {os.path.join(config.MODEL_DIR, 'training', 'weights')}")
    return results
