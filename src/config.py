from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
	# Dataset settings
	DATA_DIR:  str = "../dataset"  # Path where the dataset will be downloaded/stored
	NUM_CLASSES: int = 5  # Number of classes to work on
	IMAGES_PER_CLASS: int = 200  # amount of images for each class
	DATA_URL: str = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
	BATCH_SIZE: int = 32  # Number of images processed at once

	# Model settings
	MODEL_NAME: str = "efficientnet_b0"
	IMAGE_SIZE: int = 224  # Expected input resolution for the model
	NC: int = 3  # Number of input channels (3 = RGB Color)
	MODEL_DIR: str = "../model"  # Model after Training
	MODEL_FILE = "model.YOLO"  # Filename for the model

	# Paths
	OUT_DIR: str = "../output"  # Directory where results will be saved
	DATA_FILE = "inference.data"  # Inference Data

 
	# Plot settings
	PLOT_DPI: int = 220 	# DPI for saved plot images
	DISPLAY_PLOT: bool = True   	# Whether to display plots after execution
	HOVER_PLOT: bool = True	# Whether to show image thumbnails on hover in scatter plots
	THUMBNAIL_SIZE: int = 64	# Size of the stored hover thumbnails


@dataclass
class CLIConfig:
	inference_flags: List[str] = field(default_factory=lambda:["-i", "--inference"])
	pre_training_inference_flags: List[str] = field(default_factory=lambda:["--pre-training-inference"])
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