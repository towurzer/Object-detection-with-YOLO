# YOLO
The goal of this assignment is to implement a complete object detection pipeline that:
+ runs inference with a pre-trained YOLO model
+ visualizes and analyzes the detections (bounding boxes, class labels, confidence, scores)
+ fine-tunes (transfer learning) the YOLO model on a custom annotated dataset
+ and evaluates and compares detection performance before and after fine-tuning using standard metrics


## Project Structure
```text
src/
    config.py               # Settings
    dataset.py              # Data Pipeline: Downloads and filters the Dataset
    main.py                 # Manages the whole pipeline 
    model.py                # model
    model_trainer.py        # trains the model
    evalutaion.py           # Performs evalutaion
    utils.py                # utility functions
---
dataset/                    # Image source files
model/                      # trained model
output/                     # Saved plots and inferred data
results/                    # Results to display in README
```

## Getting Started
### 1. Installation

Run
```bash
pip install -r requirements.txt
```
to install neccessary requirements.

### 2. Run the pipeline

To reproduce the results, run the pipeline:

```bash
cd src/
python main.py
```

#### Available arguments

You can control different stages of the pipeline using the following flags:

```bash
python main.py [OPTIONS]
```

| Flag | Long option                 | Description                                                                                  |
|------|-----------------------------|----------------------------------------------------------------------------------------------|
| `-i` | `--inference`               | Run inference (used for both pre- and post-training )                                        |
|      | `--pre-training-inference`  | Run inference before training                                                                |
|      | `--post-training-inference` | Run inference after training                                                                 |
| `-t` | `--train`                   | Run the training step                                                                        |
| `-e` | `--evaluate`                | Evaluate the results                                                                         |
| `-r` | `--reset`                   | Reset all caches (including the dataset) and exit <br/> (destructive, requires confirmation) |
| `-h` | `--help`                    | Show help message and exit                                                                   |


## Results:

