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

    help_string = """
Usage:
  python main.py [OPTIONS]

Description:
  Run the full ML pipeline, including optional inference, training, and evaluation steps.
  If a step is not explicitly requested via flags, the pipeline will attempt to use cached artifacts \
(e.g., trained models, inference outputs). If required artifacts are not found, the program will exit with an error.

Options:
-i, --inference					Run inference (applies to both pre- and post-training)
--pre-training-inference		Run inference before training
--post-training-inference		Run inference after training
-t, --train						Run the training
-e, --evaluate					Evaluate the results
-r, --reset						Reset all caches (delete cached models, inference outputs, dataset, etc.) and exit
-h, --help						Show this help message and exit

Examples:
  Run training only: 'python main.py --train'   
  Run only evaluation: 'python main.py --evaluate'
  Run full pipeline: 'python main.py -i -t -e'
"""

