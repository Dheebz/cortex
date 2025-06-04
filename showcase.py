import os
import random
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# --- Global Matplotlib Style: Dark Theme ---
plt.rcParams.update(
    {
        "axes.edgecolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "figure.facecolor": "black",
        "axes.labelcolor": "white",
        "savefig.facecolor": "black",
    }
)


# --- Utility to get most recent model directory ---
def get_latest_model_dir(base_dir):
    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No model directories found in {base_dir}")
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir


# --- Paths ---
CLASSIFIER_MODEL_DIR = get_latest_model_dir("models/classifier")
DIAGNOSIS_MODEL_DIR = get_latest_model_dir("models/diagnosis")

CLASSIFIER_MODEL_PATH = os.path.join(CLASSIFIER_MODEL_DIR, "cortex-classifier.keras")
DIAGNOSIS_MODEL_PATH = os.path.join(
    DIAGNOSIS_MODEL_DIR, "cortex-classifier.keras"
)  # note: name reused

CLASSIFIER_CLASSES_PATH = os.path.join(
    CLASSIFIER_MODEL_DIR, "cotex-classifier-class_indices.json"
)
DIAGNOSIS_CLASSES_PATH = os.path.join(
    DIAGNOSIS_MODEL_DIR, "cotex-diagnosis-class_indices.json"
)

TEST_IMAGE_ROOT = "data/test"

# --- Load Models ---
classifier_model = load_model(CLASSIFIER_MODEL_PATH)
diagnosis_model = load_model(DIAGNOSIS_MODEL_PATH)

# --- Load Labels ---
with open(CLASSIFIER_CLASSES_PATH) as f:
    classifier_labels = {v: k for k, v in json.load(f).items()}
with open(DIAGNOSIS_CLASSES_PATH) as f:
    diagnosis_labels = {v: k for k, v in json.load(f).items()}


# --- Preprocessing ---
def preprocess_image(image_path, target_size=(149, 149)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)


# --- Load Images ---
image_files = [
    os.path.join(root, f)
    for root, _, files in os.walk(TEST_IMAGE_ROOT)
    for f in files
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
if not image_files:
    raise FileNotFoundError("No test images found.")


# --- Showcase with matplotlib ---
class ImageShowcase:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 10))
        self.gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.5])
        self.ax_img = self.fig.add_subplot(self.gs[0])
        self.ax_text = self.fig.add_subplot(self.gs[1])
        self.ax_text.axis("off")

        self.fig.patch.set_facecolor("black")
        self.ax_img.set_facecolor("black")
        self.ax_text.set_facecolor("black")

        # Image navigation state
        self.current_index = -1
        self.shuffled_images = image_files[:]
        random.shuffle(self.shuffled_images)
        self.prediction_cache = {}

        # Metrics
        self.total = 0
        self.correct = 0
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0

        # Auto play state
        self.autoplay_enabled = False
        self.timer = self.fig.canvas.new_timer(interval=1000)
        self.timer.add_callback(self.autoplay_step)

        # Model Version Info
        self.classifier_model_name = os.path.basename(CLASSIFIER_MODEL_DIR)
        self.diagnosis_model_name = os.path.basename(DIAGNOSIS_MODEL_DIR)

        self.show_next(1)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show()

    def autoplay_step(self):
        if self.autoplay_enabled:
            self.show_next(1)
            self.timer.start()

    def show_next(self, direction=1):
        self.current_index += direction
        self.current_index = max(
            0, min(len(self.shuffled_images) - 1, self.current_index)
        )
        image_path = self.shuffled_images[self.current_index]

        if image_path in self.prediction_cache:
            result = self.prediction_cache[image_path]
        else:
            input_image = preprocess_image(image_path)

            t0 = time.perf_counter()
            classifier_raw = classifier_model.predict(input_image, verbose=0)[0]
            classifier_time = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            diagnosis_raw = diagnosis_model.predict(input_image, verbose=0)[0]
            diagnosis_time = (time.perf_counter() - t1) * 1000

            classifier_pred_idx = np.argmax(classifier_raw)
            diagnosis_pred_idx = np.argmax(diagnosis_raw)

            classifier_pred = classifier_labels[classifier_pred_idx]
            diagnosis_pred = diagnosis_labels[diagnosis_pred_idx]
            actual_diagnosis = os.path.basename(os.path.dirname(image_path))
            actual_classifier = (
                "normal" if actual_diagnosis == "notumor" else "abnormal"
            )

            if classifier_pred == actual_classifier:
                if classifier_pred == "normal":
                    result_tag = "TN"
                    result_color = "green"
                else:
                    result_tag = "TP"
                    result_color = "green"
            else:
                if actual_classifier == "normal":
                    result_tag = "FP"
                    result_color = "yellow"
                else:
                    result_tag = "FN"
                    result_color = "red"

            result = {
                "classifier_pred": classifier_pred,
                "diagnosis_pred": diagnosis_pred,
                "actual_classifier": actual_classifier,
                "actual_diagnosis": actual_diagnosis,
                "classifier_raw": classifier_raw,
                "diagnosis_raw": diagnosis_raw,
                "classifier_time_ms": classifier_time,
                "diagnosis_time_ms": diagnosis_time,
                "result_tag": result_tag,
                "result_color": result_color,
            }
            self.prediction_cache[image_path] = result

            self.total += 1
            if classifier_pred == actual_classifier:
                self.correct += 1
                if classifier_pred == "normal":
                    self.true_negative += 1
                else:
                    self.true_positive += 1
            else:
                if actual_classifier == "normal":
                    self.false_positive += 1
                else:
                    self.false_negative += 1

        # --- Display Image ---

        self.ax_img.clear()
        self.ax_img.set_title(
            # f"Image {self.current_index + 1} of {len(self.shuffled_images)}\n"
            f"Classifier: {result['classifier_pred']} | Actual: {result['actual_classifier']}\n",
            # f"Diagnosis: {result['diagnosis_pred']} | actual: {result['actual_diagnosis']}",
            ha="center",
            va="bottom",
            fontsize=12,
            pad=20,
            color="white",
        )

        image = Image.open(image_path).convert("RGB").resize((512, 512))
        self.ax_img.imshow(image)
        self.ax_img.axis("off")

        # Result box
        box_x, box_y = 0.95, 0.05
        self.ax_img.add_patch(
            patches.Rectangle(
                (box_x - 0.08, box_y - 0.05),
                0.08,
                0.05,
                transform=self.ax_img.transAxes,
                facecolor=result["result_color"],
                edgecolor="black",
                linewidth=1,
            )
        )
        self.ax_img.text(
            box_x - 0.04,
            box_y - 0.025,
            result["result_tag"],
            transform=self.ax_img.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            weight="bold",
        )

        # --- Text Panel ---
        self.ax_text.clear()
        self.ax_text.axis("off")

        def format_probs(probs, label_map):
            return " | ".join(
                [f"{label_map[i]}: {probs[i]:.2f}" for i in range(len(probs))]
            )

        text_lines = [
            f"Processed: {self.total} | Accuracy: {(self.correct / self.total) * 100:.2f}%",
            f"TP: {self.true_positive} | TN: {self.true_negative} | FP: {self.false_positive} | FN: {self.false_negative}",
            f"File: {os.path.basename(image_path)}",
            f"Auto Play: {'ON' if self.autoplay_enabled else 'OFF'}",
            "",
            f"Classifier Time: {result['classifier_time_ms']:.2f} ms",
            f"Diagnosis Time:  {result['diagnosis_time_ms']:.2f} ms",
            "",
            f"Classifier Raw: {format_probs(result['classifier_raw'], classifier_labels)}",
            f"Diagnosis Raw:  {format_probs(result['diagnosis_raw'], diagnosis_labels)}",
            "",
            f"Classifier Model: {self.classifier_model_name}",
            f"Diagnosis Model: {self.diagnosis_model_name}",
            "",
            "[a] autoplay | [space]/→ next | ← previous | q quit",
        ]
        full_text = "\n".join(text_lines)
        self.ax_text.text(
            0.01,
            1.0,
            full_text,
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
            color="white",
        )

        self.fig.tight_layout()
        self.fig.canvas.draw()

    def on_key(self, event):
        # Stop autoplay on any key
        self.autoplay_enabled = False
        self.timer.stop()

        if event.key == " " or event.key == "right":
            self.show_next(1)
        elif event.key == "left":
            self.show_next(-1)
        elif event.key.lower() == "q":
            plt.close(self.fig)
        elif event.key.lower() == "a":  # start autoplay
            self.autoplay_enabled = True
            self.timer.start()


# --- Run ---
if __name__ == "__main__":
    ImageShowcase()
