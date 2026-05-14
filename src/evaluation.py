import os

from ultralytics import YOLO
from model import load_pretrained_model, load_trained_model

def evaluate(config):
    pre_trained_model = load_pretrained_model(config)
    trained_model = load_trained_model(config)

    image_dir = os.path.join(config.DATA_DIR, "test", "images")

    if not os.path.exists(image_dir):
        raise FileNotFoundError("No test/images folder found.")

    pre_metrics = perform_evaluation(config, pre_trained_model)
    trained_metrics = perform_evaluation(config, trained_model)

    return results_to_json(pre_metrics, trained_metrics).strip()




def perform_evaluation(config, model):
    metrics = model.val(data=os.path.join(config.DATA_DIR, config.DATASET_YAML), split='test')

    results = {
        "overall": {
            "recall": float(metrics.results_dict['metrics/recall(B)']),
            "precision": float(metrics.results_dict['metrics/precision(B)']),
            "mAP50": float(metrics.results_dict['metrics/mAP50(B)']),
            "mAP50_95": float(metrics.results_dict['metrics/mAP50-95(B)'])
        },
        "per_class": {}
    }

    # Populate per-class metrics
    for i, class_idx in enumerate(metrics.ap_class_index):
        class_name = model.names[class_idx]
        class_res = metrics.class_result(i)

        results["per_class"][class_name] = {
            "precision": float(class_res[0]),
            "recall": float(class_res[1]),
            "mAP50": float(class_res[2]),
            "mAP50_95": float(class_res[3])
        }

    return results


import json

def results_to_json(pre_trained_metrics=None, fine_tuned_metrics=None):
    combined_data = {}

    if pre_trained_metrics is not None:
        combined_data["pre_trained_baseline"] = pre_trained_metrics

    if fine_tuned_metrics is not None:
        combined_data["fine_tuned_model"] = fine_tuned_metrics

    if not combined_data:
        return json.dumps({"error": "No results provided"}, indent=4)

    return json.dumps(combined_data, indent=4)




