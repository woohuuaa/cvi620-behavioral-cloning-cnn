# Behavioral Cloning Steering CNN

A TensorFlow CNN for steering-angle prediction and autonomous driving in the
Udacity self-driving car simulator.

This project uses behavioral cloning to learn a mapping from center-camera
images to steering commands. The training pipeline follows the NVIDIA
end-to-end CNN architecture required by the CVI620 final project specification.

## Features

- Center-camera image and steering-angle dataset loading
- Steering-distribution visualization and balancing
- Random pan, zoom, rotation, brightness, and horizontal-flip augmentation
- Crop, YUV conversion, Gaussian blur, resize, and normalization
- Memory-efficient Keras batch generator
- NVIDIA CNN with validation, checkpointing, learning-rate reduction, and early stopping
- Native Windows GPU training with TensorFlow 2.10
- Socket.IO integration with the simulator's autonomous mode

## Project Structure

```text
.
|-- training.py
|-- TestSimulation.py
|-- environment-windows-gpu.yml
|-- README.md
|-- driving_log.csv               # Local dataset, not committed
|-- IMG/                          # Local camera images, not committed
|-- model.h5                      # Generated trained model, not committed
|-- simulator-windows-64/         # Local simulator, not committed
`-- training_outputs/             # Generated plots and logs
```

## Windows GPU Environment

The provided Conda environment uses:

- Python 3.9
- TensorFlow 2.10.1
- CUDA Toolkit 11.2
- cuDNN 8.1

Create and activate the environment:

```powershell
conda env create -f environment-windows-gpu.yml
conda activate CVIS26_Final
```

Verify that TensorFlow detects the GPU:

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Dataset

Place the simulator-generated data in the project root:

```text
FinalProject/
|-- driving_log.csv
`-- IMG/
```

The CSV columns are:

```text
center, left, right, steering, throttle, brake, speed
```

Only the center-camera path and steering value are used by this project.

## Preprocessing

Training preprocessing intentionally matches `TestSimulation.py`:

1. Crop rows `60:135`
2. Convert RGB to YUV
3. Apply a `3x3` Gaussian blur
4. Resize to `200x66`
5. Normalize pixel values by dividing by `255`

## Model Architecture

```text
Input: 66x200x3
Conv2D: 24 filters, 5x5, stride 2
Conv2D: 36 filters, 5x5, stride 2
Conv2D: 48 filters, 5x5, stride 2
Conv2D: 64 filters, 3x3
Conv2D: 64 filters, 3x3
Flatten
Dense: 1164
Dense: 100
Dense: 50
Dense: 10
Output: 1 steering value
```

The model uses ELU activations, Adam, mean squared error loss, and mean
absolute error as an additional metric.

## Training

Start training with the default settings:

```powershell
python training.py --epochs 20 --batch-size 64
```

Useful options:

```powershell
# Disable augmentation
python training.py --epochs 20 --batch-size 64 --no-augmentation

# Keep up to 400 samples in each steering bin
python training.py --epochs 20 --batch-size 64 --max-samples-per-bin 400

# Disable steering-bin balancing
python training.py --epochs 20 --batch-size 64 --max-samples-per-bin 0

# Show every available option
python training.py --help
```

Default settings:

```text
epochs                  20
batch size              64
validation split        0.20
steering bins           31
max samples per bin     200
augmentation            enabled
random seed             42
```

Each run writes:

```text
model.h5
training_outputs/steering_histogram.png
training_outputs/training_history.png
training_outputs/training_log.csv
```

These files are overwritten by later runs, so preserve experiment artifacts
before starting another configuration.

## Autonomous-Mode Testing

Run the server from the project directory:

```powershell
conda activate CVIS26_Final
cd C:\path\to\behavioral-cloning-steering-cnn
python TestSimulation.py
```

Keep that terminal open. In a second terminal, launch the simulator:

```powershell
& ".\simulator-windows-64\Default Windows desktop 64-bit.exe"
```

Select `AUTONOMOUS MODE`. The simulator connects automatically to the
Socket.IO server on `localhost:4567`.

Successful communication prints:

```text
Connected
```

followed by live throttle, steering, and speed values.

## Notes

- Training augmentation is never applied to validation data.
- The validation split is used only for evaluation and does not update model weights.
- `EarlyStopping` restores the best validation weights.
- `model.h5`, the simulator, and the recorded dataset are excluded from Git.
- If a pre-trained model is distributed separately, place it in the project root before running `TestSimulation.py`.
