# Pedestrian Detection on Raspberry PI
This is a project with the aim of integrating a pedestrian detection model in a Raspberry PI

- Sami Carret (sami.carret@epita.fr)
- Florine Kieraga (florine.kieraga@epita.fr)
- Samy Amine (samy.amine@epita.fr)


## Quantization

To get started, we first need to preprocess our model weights. After exporting the base model to onnx, run the following:

```bash
python -m onnxruntime.quantization.preprocess --input best.onnx --output best_preprocessed.onnx
```

https://medium.com/@sulavstha007/quantizing-yolo-v8-models-34c39a2c10e2


http://craft@raspberrypi.local:5000/video_feed
