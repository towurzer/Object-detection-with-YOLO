import os
from roboflow import Roboflow
from dotenv import load_dotenv
load_dotenv()


def download_dataset(config):
    """
    Automatically downloads the configured dataset from Roboflow
    if it does not already exist locally.
    """
    # Skip download if dataset already exists
    if os.path.exists(os.path.join(config.DATA_DIR, config.DATASET_YAML)):
        print("Dataset found, downloading skipped.")
        return

    print("Downloading dataset from Roboflow...")

    # API key is read from environment variable
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ROBOFLOW_API_KEY environment variable not set.\n"
            "Get your API key at: https://app.roboflow.com/settings/api\n"
            "Linux/macOS current session: export ROBOFLOW_API_KEY='your_key_here'\n"
            "Windows PowerShell current session: $env:ROBOFLOW_API_KEY='your_key_here'\n"
            )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(config.ROBOFLOW_WORKSPACE).project(config.ROBOFLOW_PROJECT)
    version = project.version(config.ROBOFLOW_VERSION)
    version.download("yolov8", location=config.DATA_DIR, overwrite=True)


    print(f"Dataset downloaded to: {config.DATA_DIR}")


def get_dataloader(config):
    """
    Downloads dataset if needed, validates structure, and returns
    path to data.yaml for use with YOLO training and inference.
    """
    # Download dataset if not present
    download_dataset(config)

    # Build path to data.yaml
    data_yaml = os.path.join(config.DATA_DIR, config.DATASET_YAML)

    # Check that data.yaml exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"{config.DATASET_YAML} not found in '{config.DATA_DIR}'\n"
            f"Make sure the dataset is in YOLOv8 format."
        )

    # Check that train/valid splits exist
    for split in ["train", "valid"]:
        split_path = os.path.join(config.DATA_DIR, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Expected folder '{split}' not found in '{config.DATA_DIR}'"
            )

    # Test split is optional
    if not os.path.exists(os.path.join(config.DATA_DIR, "test")):
        print("Warning: 'test' folder not found - evaluation in Step 5 may not work.")

    print(f"Dataset found: {config.DATA_DIR}")
    print(f"Using config: {data_yaml}")

    return data_yaml