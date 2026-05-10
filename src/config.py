from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    # Dataset settings
    DATA_DIR: str = str(PROJECT_ROOT / "dataset")
    NUM_CLASSES: int = 4
    IMAGES_PER_CLASS: int = 200
    DATA_URL: str = "https://data.caltech.edu/records/mzrjg-6wc02/files/caltech-101.zip"
    BATCH_SIZE: int = 32
    DATA_YAML: str = "data.yaml"

    # Model settings
    MODEL_NAME: str = "yolo11n.pt"
    IMAGE_SIZE: int = 640
    NC: int = 3
    MODEL_DIR: str = str(PROJECT_ROOT / "model")
    MODEL_FILE: str = "model.YOLO"
    CONF_THRESHOLD: float = 0.25

    # Paths
    OUT_DIR: str = str(PROJECT_ROOT / "output")
    DATA_FILE: str = "inference.data"

    # Plot settings
    PLOT_DPI: int = 220
    DISPLAY_PLOT: bool = True
    HOVER_PLOT: bool = True
    THUMBNAIL_SIZE: int = 64

@dataclass
class CLIConfig:
    inference_flags: List[str] = field(default_factory=lambda: ["-i", "--inference"])
    pre_training_inference_flags: List[str] = field(default_factory=lambda: ["--pre-training-inference"])
    training_flags: List[str] = field(default_factory=lambda: ["-t", "--train"])
    post_training_inference_flags: List[str] = field(default_factory=lambda: ["--post-training-inference"])
    evaluate_flags: List[str] = field(default_factory=lambda: ["-e", "--evaluate"])
    reset_flags: List[str] = field(default_factory=lambda: ["-r", "--reset"])
    help_flags: List[str] = field(default_factory=lambda: ["-h", "--help"])

    help_string = ""

