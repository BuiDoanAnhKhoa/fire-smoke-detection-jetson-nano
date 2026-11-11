# What is YAML, and why are there two dataset configurations?

YAML is a human-readable text format for configuration. In YOLO training it tells
the library **where the data is** and **what the class IDs mean**. It contains
settings, not images, annotations, or neural-network weights. Indentation matters.

## Original split: data.yaml

```yaml
path: /absolute/path/to/HomeFire
train: train/images
val: val/images
test: test/images
names:
  0: fire
  1: smoke
```

`path` is the dataset root. The other three paths select directories below it.
Images physically in `train/images` are training examples, and so on. YOLO finds
their corresponding text labels in the matching `labels` directories.

This describes the original split used by the Colab script. It is retained for
comparison and provenance.

## Local training split: data_grouped.yaml

```yaml
path: /absolute/path/to/HomeFire
train: /absolute/path/to/HomeFire/splits/train.txt
val: /absolute/path/to/HomeFire/splits/val.txt
test: /absolute/path/to/HomeFire/splits/test.txt
names:
  0: fire
  1: smoke
```

Here each `.txt` file is an **image list**, with one image path per line. These
lists decide membership rather than the original folder name. They are distinct
from the per-image annotation `.txt` files containing bounding boxes.

We detected visually similar frames across original splits and created new
groups to reduce this overlap. Images in each detected group stay in one split.
We preserved the original image/label files and changed membership through lists.
The completed local run used **data_grouped.yaml**, with 3,900 training, 1,300
validation, and 1,300 test images. Grouping is heuristic and does not guarantee
complete separation between source videos.

## Which file do I use?

- For explaining the completed local training: **data_grouped.yaml**.
- For explaining the original Colab setup: **data.yaml**.
- For laptop inference with **best.pt**: **neither**.

The checkpoint contains the model and its class mapping. `predict_fire.py` needs
the checkpoint plus an image/video/camera source. These two configuration copies
use placeholder paths for explanation and cannot train without the omitted data.

There is also `results/args.yaml`: that is the recorded full **training settings**
(batch size, learning rate, image size, and other options), not a dataset split file.
