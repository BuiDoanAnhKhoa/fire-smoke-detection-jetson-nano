# Jetson Nano deployment roadmap

Status: planned. The desktop model is trained and evaluated. The target Jetson
Nano 2GB has an OS installed; this repository does not yet establish camera or
TensorRT operation on that device.

1. Record the installed L4T/JetPack, CUDA, TensorRT, and Python versions.
2. Confirm live capture from the IMX219-160 camera before integrating inference.
3. Export the trained checkpoint to static-shape, batch-one ONNX using an opset
   supported by the installed TensorRT parser.
4. Compare PyTorch and ONNX predictions on identical inputs and preprocessing.
5. Build the TensorRT engine on the Nano. Measure FP16 behavior and memory usage.
6. Implement camera capture, letterboxing, normalization, inference, NMS, and
   coordinate restoration. Verify results against the desktop reference.
7. Compare 320, 416, and 640 input sizes using target-camera validation footage.
8. Measure end-to-end latency, sustained FPS, RAM, temperature, false alarms, and
   smoke misses. Keep the camera queue bounded to avoid processing stale frames.
9. Integrate timestamped ESP32/MQTT readings from MQ-2, DHT11, and IR flame sensors.
   Distinguish missing/stale readings from a confirmed absence of detections.
10. Evaluate time to warning, event recall, and false alarms per hour for both
    AI-only and fused decisions.

Initial read-only checks on the Nano:

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack libnvinfer8
free -h
gst-inspect-1.0 nvarguscamerasrc
```

Desktop evaluation latency is not a Jetson FPS result. Exact install/export
commands should be chosen after confirming the board's installed software.
