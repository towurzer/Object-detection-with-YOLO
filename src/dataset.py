import os

def get_dataloader(config):
    """
    Validates the dataset directory and returns the path to data.yaml.
    YOLO uses the data.yaml file directly for training and inference.

    :param config: Configuration object containing DATA_DIR path.
    :return: Absolute path to data.yaml file.
    """

    # Build path to data.yaml
    data_yaml = os.path.join(config.DATA_DIR, "data.yaml")

    # Check that dataset folder exists
    if not os.path.exists(config.DATA_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found: '{config.DATA_DIR}'\n"
            f"Please place the dataset in the '{config.DATA_DIR}' folder."
        )

    # Check that data.yaml exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"data.yaml not found in '{config.DATA_DIR}'\n"
            f"Make sure the dataset is in YOLOv8 format."
        )

    # Check that train/valid/test folders exist
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(config.DATA_DIR, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Expected folder '{split}' not found in '{config.DATA_DIR}'"
            )

    print(f"Dataset found: {config.DATA_DIR}")
    print(f"Using config: {data_yaml}")

    return data_yaml