"""Train the steering-angle CNN required by Final_Project.pdf.

Run:
    python training.py --epochs 20 --batch-size 64
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

matplotlib.use("Agg")
from matplotlib import pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DRIVING_LOG_PATH = PROJECT_DIR / "driving_log.csv"
IMAGE_DIR = PROJECT_DIR / "IMG"
MODEL_PATH = PROJECT_DIR / "model.h5"
OUTPUT_DIR = PROJECT_DIR / "training_outputs"

PAN_RANGE_X = 40.0
PAN_RANGE_Y = 8.0
PAN_STEERING_FACTOR = 0.002
ROTATION_RANGE_DEGREES = 5.0
ROTATION_STEERING_FACTOR = 0.01

# Speed normalization divisor. Must match the value used by TestSimulation.py
# when it builds the speed input at inference time.
MAX_SPEED = 30.0

# Number of consecutive rows kept together on one side of the train/validation
# split. Splitting individual frames would let near-duplicate consecutive
# frames (captured ~every 100ms) leak across the split, making validation
# loss look better than the model's real generalization.
VALIDATION_CHUNK_SIZE = 50


def load_rgb_image(image_path):
    """Load an image with OpenCV and convert it from BGR to RGB."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def preProcessing(img):
    """Apply the exact preprocessing used by TestSimulation.py."""
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255

    return img


def _warp_image(image, matrix):
    """Apply an affine transform without changing the image dimensions."""
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def augment_image(image, steering, rng):
    """Randomly apply pan, zoom, rotation, brightness, and horizontal flip."""
    height, width = image.shape[:2]
    augmented = image.copy()
    steering = float(steering)

    # Pan the image and adjust the steering angle accordingly
    if rng.random() < 0.5:
        shift_x = float(rng.uniform(-PAN_RANGE_X, PAN_RANGE_X))
        shift_y = float(rng.uniform(-PAN_RANGE_Y, PAN_RANGE_Y))
        pan_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        augmented = _warp_image(augmented, pan_matrix)
        steering += shift_x * PAN_STEERING_FACTOR

    # Zoom the image and adjust the steering angle accordingly
    if rng.random() < 0.5:
        zoom = float(rng.uniform(1.0, 1.2))
        zoom_matrix = cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0),
            0.0,
            zoom,
        )
        augmented = _warp_image(augmented, zoom_matrix)

    # Rotate the image and adjust the steering angle accordingly
    if rng.random() < 0.5:
        # Randomly choose a rotation angle between -5 and 5
        angle = float(rng.uniform(-ROTATION_RANGE_DEGREES, ROTATION_RANGE_DEGREES))

        # Create a rotation matrix around the center of the image
        rotation_matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0,)
        augmented = _warp_image(augmented, rotation_matrix)

        # Adjust the steering angle based on the rotation angle
        steering -= angle * ROTATION_STEERING_FACTOR

    # Adjust the brightness of the image in HSV color space
    if rng.random() < 0.5:
        # Randomly choose a brightness factor between 0.6 and 1.2

        brightness = float(rng.uniform(0.6, 1.2))

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV)

        # Scale the V channel by the brightness factor and clip to [0, 255]
        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2].astype(np.float32) * brightness, 0, 255,).astype(np.uint8)

        # Convert back to RGB color space
        augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Randomly flip the image horizontally and invert the steering angle
    if rng.random() < 0.5:
        # Flip the image horizontally
        augmented = cv2.flip(augmented, 1)
        # Invert the steering angle
        steering *= -1.0

    # Clip the steering angle to the range [-1.0, 1.0] to avoid extreme values
    steering = float(np.clip(steering, -1.0, 1.0))

    # Return the augmented image and the adjusted steering angle as contiguous arrays
    return np.ascontiguousarray(augmented), steering


def load_driving_rows(
    csv_path=DRIVING_LOG_PATH,
    image_dir=IMAGE_DIR,
):
    """Load center-camera path, steering, and speed from driving_log.csv.

    Returns one entry per recorded instant, in original CSV row order, so
    callers can split train/validation by contiguous chunks.
    """
    rows = []

    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 7:
                raise ValueError(
                    f"Row {row_number} has {len(row)} columns; expected at least 7"
                )

            center_path = image_dir / Path(row[0].strip()).name
            if not center_path.is_file():
                raise FileNotFoundError(
                    f"Center image from row {row_number} was not found: {center_path}"
                )

            try:
                steering = float(row[3].strip())
            except ValueError as error:
                raise ValueError(
                    f"Invalid steering value on row {row_number}: {row[3]!r}"
                ) from error

            try:
                speed = float(row[6].strip())
            except ValueError as error:
                raise ValueError(
                    f"Invalid speed value on row {row_number}: {row[6]!r}"
                ) from error

            rows.append((center_path, steering, speed))

    if not rows:
        raise ValueError(f"No driving rows found in {csv_path}")

    return rows


def split_rows_by_chunk(rows, validation_split, chunk_size, seed):
    """Split rows into train/validation by contiguous chunks, not individual frames.

    Splitting frame-by-frame would let near-duplicate consecutive frames
    (captured ~every 100ms) end up on both sides of the split, making
    validation loss look better than the model's real generalization. Keeping
    each contiguous chunk together on one side avoids that leakage.
    """
    chunk_ids = np.arange(len(rows)) // chunk_size
    unique_chunk_ids = np.unique(chunk_ids)

    train_chunk_ids, validation_chunk_ids = train_test_split(
        unique_chunk_ids,
        test_size=validation_split,
        random_state=seed,
        shuffle=True,
    )
    validation_chunk_id_set = set(validation_chunk_ids.tolist())

    train_rows, validation_rows = [], []
    for row, chunk_id in zip(rows, chunk_ids):
        if chunk_id in validation_chunk_id_set:
            validation_rows.append(row)
        else:
            train_rows.append(row)

    return train_rows, validation_rows


def balance_training_samples(samples, bins, max_samples_per_bin, seed):
    """Limit each steering bin to reduce an oversized straight-driving peak."""
    if bins < 3:
        raise ValueError("--steering-bins must be at least 3")
    if max_samples_per_bin < 0:
        raise ValueError("--max-samples-per-bin cannot be negative")
    if max_samples_per_bin == 0:
        return list(samples)

    steering = np.asarray([sample[1] for sample in samples], dtype=np.float32)
    bin_edges = np.linspace(-1.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(steering, bin_edges[1:-1]), 0, bins - 1)
    rng = np.random.default_rng(seed)
    retained_indices = []

    for bin_id in range(bins):
        sample_indices = np.flatnonzero(bin_ids == bin_id)
        if len(sample_indices) > max_samples_per_bin:
            sample_indices = rng.choice(
                sample_indices,
                size=max_samples_per_bin,
                replace=False,
            )
        retained_indices.extend(int(index) for index in sample_indices)

    rng.shuffle(retained_indices)
    return [samples[index] for index in retained_indices]


def plot_steering_histogram(before_balancing, after_balancing, bins, filename="steering_histogram.png"):
    """Save steering histograms before and after dataset balancing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    before_steering = [sample[1] for sample in before_balancing]
    after_steering = [sample[1] for sample in after_balancing]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(
        before_steering,
        bins=bins,
        range=(-1.0, 1.0),
        color="#3B82A0",
    )
    axes[0].set_title("Before Balancing")
    axes[1].hist(
        after_steering,
        bins=bins,
        range=(-1.0, 1.0),
        color="#E07A3F",
    )
    axes[1].set_title("After Balancing")

    for axis in axes:
        axis.set_xlabel("Steering Angle")
        axis.set_ylabel("Samples")
        axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


class DrivingSequence(Sequence):
    """Load, augment, and preprocess simulator images one batch at a time."""

    def __init__(self, samples, batch_size, augment, shuffle, seed, use_speed=True):
        """Store batch settings and initialize the sample index order."""
        self.samples = samples
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.use_speed = use_speed
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(samples))
        self.on_epoch_end()

    def __len__(self):
        """Return the number of batches in one epoch."""
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, batch_index):
        """Load and return one batch of images and steering values."""
        start = batch_index * self.batch_size
        stop = min(start + self.batch_size, len(self.samples))
        batch_indices = self.indices[start:stop]

        images = np.empty((len(batch_indices), 66, 200, 3), dtype=np.float32)
        speeds = np.empty((len(batch_indices), 1), dtype=np.float32)
        steering_values = np.empty(len(batch_indices), dtype=np.float32)

        # Load, augment, and preprocess each sample in the batch
        for output_index, sample_index in enumerate(batch_indices):
            image_path, steering, speed = self.samples[int(sample_index)]
            image = load_rgb_image(image_path)

            if self.augment:
                image, steering = augment_image(image, steering, self.rng)

            images[output_index] = preProcessing(image)
            speeds[output_index] = speed / MAX_SPEED
            steering_values[output_index] = steering

        if self.use_speed:
            return {"preprocessed_image": images, "speed": speeds}, steering_values
        return images, steering_values

    def on_epoch_end(self):
        """Shuffle training sample indices after each epoch when enabled."""
        if self.shuffle:
            self.rng.shuffle(self.indices)


def build_nvidia_model_with_speed():
    """Build and compile the NVIDIA CNN (PDF Figure 7), extended with a speed input.

    The convolutional stack is unchanged from the required architecture. Current
    vehicle speed is concatenated onto the flattened image features so the same
    visual curve can map to a different steering angle depending on speed.
    """
    image_input = Input(shape=(66, 200, 3), name="preprocessed_image")
    speed_input = Input(shape=(1,), name="speed")

    x = Conv2D(24, (5, 5), strides=(2, 2), activation="elu", name="conv_1")(image_input)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation="elu", name="conv_2")(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation="elu", name="conv_3")(x)
    x = Conv2D(64, (3, 3), activation="elu", name="conv_4")(x)
    x = Conv2D(64, (3, 3), activation="elu", name="conv_5")(x)
    x = Flatten(name="flatten")(x)
    x = Concatenate(name="image_features_and_speed")([x, speed_input])
    x = Dense(1164, activation="elu", name="dense_1164")(x)
    x = Dense(100, activation="elu", name="dense_100")(x)
    x = Dense(50, activation="elu", name="dense_50")(x)
    x = Dense(10, activation="elu", name="dense_10")(x)
    steering_output = Dense(1, name="steering")(x)

    model = Model(
        inputs=[image_input, speed_input],
        outputs=steering_output,
        name="nvidia_steering_model_speed_aware",
    )
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])
    return model


def build_nvidia_model_image_only():
    """Build and compile the original NVIDIA CNN (PDF Figure 7), image-only."""
    model = Sequential([
        Input(shape=(66, 200, 3), name="preprocessed_image"),
        Conv2D(24, (5, 5), strides=(2, 2), activation="elu", name="conv_1"),
        Conv2D(36, (5, 5), strides=(2, 2), activation="elu", name="conv_2"),
        Conv2D(48, (5, 5), strides=(2, 2), activation="elu", name="conv_3"),
        Conv2D(64, (3, 3), activation="elu", name="conv_4"),
        Conv2D(64, (3, 3), activation="elu", name="conv_5"),
        Flatten(name="flatten"),
        Dense(1164, activation="elu", name="dense_1164"),
        Dense(100, activation="elu", name="dense_100"),
        Dense(50, activation="elu", name="dense_50"),
        Dense(10, activation="elu", name="dense_10"),
        Dense(1, name="steering"),
    ], name="nvidia_steering_model")

    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])
    return model


def configure_gpu():
    """Enable gradual GPU memory allocation and report the selected device."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus:
        print(f"GPU enabled: {gpus[0]}")
    else:
        print("GPU not found; training will use CPU")


def plot_training_history(history, filename="training_history.png"):
    """Save training and validation MSE/MAE curves to an image."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Mean Squared Error")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="train")
    axes[1].plot(history.history["val_mae"], label="validation")
    axes[1].set_title("Mean Absolute Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def parse_args():
    """Read training settings supplied through the command line."""
    parser = argparse.ArgumentParser(
        description="Train the PDF Figure 7 steering-angle CNN."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--steering-bins", type=int, default=31)
    parser.add_argument(
        "--max-samples-per-bin",
        type=int,
        default=200,
        help="Maximum training samples retained per steering bin; 0 disables balancing.",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable random augmentation for the training dataset.",
    )
    parser.add_argument(
        "--no-speed",
        action="store_true",
        help=(
            "Train the original image-only NVIDIA architecture instead of the "
            "speed-aware variant. Writes to model_no_speed.h5 and a separate "
            "set of training_outputs files, leaving model.h5 untouched."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """Run dataset preparation, model training, and evaluation."""
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not 0.0 < args.validation_split < 1.0:
        raise ValueError("--validation-split must be between 0 and 1")

    np.random.seed(args.seed)
    configure_gpu()
    tf.random.set_seed(args.seed)

    use_speed = not args.no_speed
    model_path = MODEL_PATH if use_speed else PROJECT_DIR / "model_no_speed.h5"
    suffix = "" if use_speed else "_no_speed"

    rows = load_driving_rows()
    # Split by contiguous chunks (not individual frames) so near-duplicate
    # consecutive frames can't leak across the train/validation split.
    train_rows, validation_rows = split_rows_by_chunk(
        rows,
        validation_split=args.validation_split,
        chunk_size=VALIDATION_CHUNK_SIZE,
        seed=args.seed,
    )

    train_samples_before_balancing = train_rows
    validation_samples = validation_rows

    # Balance the training samples
    train_samples = balance_training_samples(
        train_samples_before_balancing,
        bins=args.steering_bins,
        max_samples_per_bin=args.max_samples_per_bin,
        seed=args.seed,
    )
    if not train_samples:
        raise ValueError("Dataset balancing removed all training samples")

    # Create a histogram of the steering angles before and after balancing
    # for 4. Reviewing and Balancing the Dataset
    histogram_path = plot_steering_histogram(
        train_samples_before_balancing,
        train_samples,
        bins=args.steering_bins,
        filename=f"steering_histogram{suffix}.png",
    )

    # Create training and validation sequences for Keras model.fit()
    training_sequence = DrivingSequence(
        train_samples,
        batch_size=args.batch_size,
        augment=not args.no_augmentation,
        shuffle=True,
        seed=args.seed,
        use_speed=use_speed,
    )
    validation_sequence = DrivingSequence(
        validation_samples,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
        seed=args.seed,
        use_speed=use_speed,
    )

    # Print dataset and training information
    first_inputs, first_steering = training_sequence[0]
    print(f"Loaded rows: {len(rows)} (train rows: {len(train_rows)}, validation rows: {len(validation_rows)})")
    print(
        "Training samples before balancing: "
        f"{len(train_samples_before_balancing)}"
    )
    print(f"Training samples after balancing: {len(train_samples)}")
    print(f"Validation samples: {len(validation_samples)}")
    print(f"Training augmentation: {not args.no_augmentation}")
    print(f"Using speed input: {use_speed}")
    print(f"Saved steering histogram: {histogram_path}")
    if use_speed:
        print(
            "First batch:",
            f"images={first_inputs['preprocessed_image'].shape} {first_inputs['preprocessed_image'].dtype}",
            f"speed={first_inputs['speed'].shape} {first_inputs['speed'].dtype}",
            f"steering={first_steering.shape} {first_steering.dtype}",
        )
    else:
        print(
            "First batch:",
            f"images={first_inputs.shape} {first_inputs.dtype}",
            f"steering={first_steering.shape} {first_steering.dtype}",
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_nvidia_model_with_speed() if use_speed else build_nvidia_model_image_only()
    model.summary()

    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger(OUTPUT_DIR / f"training_log{suffix}.csv"),
    ]

    history = model.fit(
        training_sequence,
        validation_data=validation_sequence,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(model_path)
    history_path = plot_training_history(history, filename=f"training_history{suffix}.png")
    validation_loss, validation_mae = model.evaluate(
        validation_sequence,
        verbose=0,
    )

    print(f"Validation MSE: {validation_loss:.6f}")
    print(f"Validation MAE: {validation_mae:.6f}")
    print(f"Saved model: {model_path}")
    print(f"Saved history plot: {history_path}")


if __name__ == "__main__":
    main()
