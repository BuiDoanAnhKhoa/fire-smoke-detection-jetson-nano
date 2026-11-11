"""Detect fire and smoke in a webcam feed, image, directory, or video."""
import argparse
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='0', help='Webcam index, image, image directory, or video path')
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'models/best.pt')
    parser.add_argument('--device', default='cpu', help='cpu, or 0 for a compatible NVIDIA GPU')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--show', action='store_true', help='Display detections; press q in the window to quit')
    parser.add_argument('--save', action='store_true', help='Save annotated images/video')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results/predictions')
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f'Model does not exist: {args.model}')
    if not 0 <= args.conf <= 1 or args.imgsz < 32:
        parser.error('conf must be between 0 and 1, and imgsz must be at least 32')
    from ultralytics import YOLO
    # 1. Load the learned weights. This does not train the model.
    model = YOLO(str(args.model))
    if model.names != {0: 'fire', 1: 'smoke'}:
        parser.error(f'Unexpected model classes: {model.names}')
    source = int(args.source) if args.source.isdigit() else args.source
    print(f'Model: {args.model}\nClasses: {model.names}\nDevice: {args.device}\nConfidence: {args.conf}')
    name = datetime.now().strftime('predict_%Y%m%d_%H%M%S_%f')
    # 2. Process one frame at a time, keeping video memory usage bounded.
    results = model.predict(source=source, device=args.device, imgsz=args.imgsz, conf=args.conf,
                            stream=True, show=args.show, save=args.save, verbose=True,
                            project=str(args.output.resolve()), name=name, exist_ok=False)
    count = 0
    try:
        for result in results:
            count += 1
            # Ultralytics displays/saves predictions according to --show/--save.
    except KeyboardInterrupt:
        print('\nStopped by user.')
    finally:
        results.close()
    print(f'Processed {count} image(s)/frame(s).')
    if args.save:
        print(f'Annotated results: {args.output.resolve() / name}')


if __name__ == '__main__':
    main()
