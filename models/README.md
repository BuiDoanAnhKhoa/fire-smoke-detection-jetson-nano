# Model artifacts

`best.pt` is the trained two-class YOLOv8n checkpoint selected at epoch 72.
It is included so that inference works without training or downloading HomeFire.

SHA256:

```text
736629455959a482400e68fdd93d7149f2a2b65b2dfb432f9c26598108b18a1c
```

The COCO initialization `yolov8n.pt` is needed only for training and is excluded
from Git. Download the pinned artifact referenced in [SOURCE.md](SOURCE.md),
save it in this directory, and verify its recorded checksum.

Exported ONNX and TensorRT files are generated locally and are also excluded.
The current repository does not contain a verified Jetson runtime or engine.
