from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Config:    
    # Roboflow dataset configuration
    # Workspace and project on Roboflow Universe (change if you switch dataset)
    ROBOFLOW_WORKSPACE: str = "gao-shou-zheng-b6xqc"
    ROBOFLOW_PROJECT: str = "solar-panel-0swal"
    ROBOFLOW_VERSION: int = 3
    
    # Training settings
    BATCH_SIZE: int = 4
    TRAIN_EPOCHS: int = 30
    TRAIN_DEVICE: str = "cpu"

    # Model settings
    MODEL_NAME: str = "yolo11n.pt"
    IMAGE_SIZE: int = 640
    MODEL_FILE: str = "model.YOLO"
    CONF_THRESHOLD: float = 0.25

    # Paths
    DATA_DIR: str = str(PROJECT_ROOT / "dataset")
    MODEL_DIR: str = str(PROJECT_ROOT / "model")
    OUT_DIR: str = str(PROJECT_ROOT / "output")
    DATA_FILE: str = "inference.data"


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

