from detectron2.checkpoint import DetectionCheckpointer

from .load_darknet import load_darknet_weights_to_dict


class YOLOV3Checkpointer(DetectionCheckpointer):
    """
    Same as :class:`DetectionCheckpointer`, but can load official darknet weights.
    """

    def _load_file(self, filename):
        if filename.endswith(".weights"):
            with self.path_manager.open(filename, "rb") as f:
                darknet_convs = self.model.backbone.darknet_modules()
                loaded = load_darknet_weights_to_dict(f, darknet_convs)
        else:
            loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded
