from ultralytics import YOLO

class Model:
    def __init__(self, model_name):
        self.model = YOLO(model_name)

    def get_model(self):
        return self.model
