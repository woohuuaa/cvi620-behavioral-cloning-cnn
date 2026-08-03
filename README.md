# Behavioral Cloning Steering CNN

A TensorFlow CNN for steering-angle prediction and autonomous driving in the
Udacity self-driving car simulator.

This project uses behavioral cloning to learn a mapping from center-camera
images to steering commands. The training pipeline follows the NVIDIA
end-to-end CNN architecture required by the CVI620 final project specification.

## Features

- Center-camera image and steering-angle dataset loading
- Steering-distribution visualization and balancing
- Mild random brightness and zoom augmentation
- Contiguous-chunk train/validation splitting to reduce frame leakage
- Crop, YUV conversion, Gaussian blur, resize, and normalization
- Memory-efficient Keras batch generator
- NVIDIA CNN with validation, checkpointing, learning-rate reduction, and early stopping
- Native Windows GPU training with TensorFlow 2.10
- macOS training via TensorFlow-Metal (Apple Silicon)
- Socket.IO integration with the simulator's autonomous mode

## Project Structure

```text
.
|-- training.py
|-- TestSimulation.py
|-- environment-windows-gpu.yml
|-- environment-mac.yml
|-- README.md
|-- driving_log.csv               # Local dataset, not committed
|-- IMG/                          # Local camera images, not committed
|-- model.h5                      # Generated trained model, not committed
|-- simulator-windows-64/         # Local simulator, not committed
|-- simulator-mac/                # Local simulator, not committed
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

## macOS Environment (Apple Silicon)

The provided Conda environment mirrors the Windows environment, swapping the
CUDA/cuDNN GPU stack for Apple's Metal backend:

- Python 3.9
- TensorFlow 2.10.0 (`tensorflow-macos` + `tensorflow-metal`)
- All other pinned package versions match `environment-windows-gpu.yml`, except
  `numpy` (1.23.5 instead of 1.22.4, as required by the
  `tensorflow-macos` 2.10.0 wheel's compiled NumPy ABI)

Install [Miniforge](https://github.com/conda-forge/miniforge) if you don't
already have Conda:

```bash
brew install --cask miniforge
conda init zsh
```

Create and activate the environment:

```bash
conda env create -f environment-mac.yml
conda activate CVIS26_Final
```

Verify that TensorFlow detects the GPU:

```bash
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

Only the center-camera path and steering value are used for training. The
left-camera, right-camera, throttle, brake, and speed columns remain in the
simulator-generated CSV but are not used by the model.

## Dataset Splitting and Balancing

Consecutive simulator frames are highly similar because they are recorded at
short time intervals. Splitting individual frames randomly could place nearly
identical images in both the training and validation sets, making validation
results overly optimistic.

To reduce this leakage, the pipeline groups every 50 consecutive rows into a
chunk and assigns each complete chunk to either training or validation. By
default, approximately 80% of the chunks are used for training and 20% for
validation.

Steering-bin balancing is applied only to the training set. Validation data is
kept unchanged and contains only the original center-camera samples.

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

The model follows the image-only NVIDIA end-to-end CNN architecture required
by the project specification. It uses ELU activations, the Adam optimizer with
an initial learning rate of `0.0001`, mean squared error loss, and mean absolute
error as an additional metric.

## Augmentation

Augmentation is applied only to training samples. Each sample has a 70% chance
of remaining unchanged. Otherwise, exactly one mild transformation is selected:

- 15% overall chance of brightness adjustment in the range `0.8` to `1.2`
- 15% overall chance of center-based zoom in the range `1.0` to `1.08`

These transformations do not change the steering label. Pan, rotation, and
horizontal flipping are intentionally not used because they produced less
representative samples for the mostly smooth, one-direction track.

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
max samples per bin     400
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

On macOS, run the equivalent commands:

```bash
conda activate CVIS26_Final
cd /path/to/behavioral-cloning-steering-cnn
python TestSimulation.py
```

Keep that terminal open. In a second terminal, launch the simulator app:

```bash
open "simulator-mac/Default Mac desktop Universal.app"
```

The simulator app is unsigned, so the first launch requires clearing macOS's
quarantine flag (one-time):

```bash
xattr -dr com.apple.quarantine "simulator-mac/Default Mac desktop Universal.app"
```

Successful communication prints:

```text
Connected
```

followed by live throttle, steering, and speed values.

## Notes

- Training augmentation is never applied to validation data.
- Dataset balancing is never applied to validation data.
- Consecutive rows are split in chunks of 50 to reduce leakage between training and validation.
- The validation split is used only for evaluation and does not update model weights.
- `EarlyStopping` restores the best validation weights.
- `model.h5`, the simulator, and the recorded dataset are excluded from Git.
- If a pre-trained model is distributed separately, place it in the project root before running `TestSimulation.py`.
