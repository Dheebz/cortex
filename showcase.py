import os
import random
import json
import time
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model

# Load label maps
with open('models/cnn/cortex-cnn-model/cortex-cnn-lvl-1_labels.json') as f:
    lvl1_labels = json.load(f)

with open('models/cnn/cortex-cnn-model/cortex-cnn-lvl-2_labels.json') as f:
    lvl2_labels = json.load(f)

lvl1_inv = {v: k for k, v in lvl1_labels.items()}
lvl2_inv = {v: k for k, v in lvl2_labels.items()}

# Load models
model_lvl1 = load_model('models/cnn/cortex-cnn-model/cortex-cnn-lvl-1.keras')
model_lvl2 = load_model('models/cnn/cortex-cnn-model/cortex-cnn-lvl-2.keras')
_ = model_lvl1(np.zeros((1, 149, 149, 3)))
_ = model_lvl2(np.zeros((1, 149, 149, 3)))

# Valid conv layers for Grad-CAM
valid_conv_layers = [l.name for l in model_lvl2.layers if isinstance(l, tf.keras.layers.Conv2D)]

# Grad-CAM

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0] * pooled_grads
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return tf.image.resize(heatmap[..., tf.newaxis], (149, 149)).numpy().squeeze()

def overlay_gradcam(img, heatmap, alpha=0.5):
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_img = Image.fromarray((jet_heatmap * 255).astype(np.uint8)).resize(img.size)
    return Image.blend(img.convert('RGBA'), jet_img.convert('RGBA'), alpha)

# UI
st.set_page_config(layout="wide")
st.title("🧠 Cortex Showcase Viewer")

mode = st.sidebar.selectbox("Select Mode", ["Level 1 Only", "Level 2 Only", "Combined"])
view_mode = st.sidebar.radio("View Mode", ["Standard", "Grad-CAM", "Both"])
selected_layer = st.sidebar.selectbox("Select Conv Layer for Grad-CAM", valid_conv_layers)
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if 'image_paths' not in st.session_state:
    image_dir = 'data/test'
    st.session_state.image_paths = [os.path.join(dp, f) for dp, _, files in os.walk(image_dir) for f in files if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    random.shuffle(st.session_state.image_paths)

if 'index' not in st.session_state:
    st.session_state.index = 0

if uploaded_file:
    img_name = uploaded_file.name
    img = Image.open(uploaded_file).convert('RGB').resize((149, 149))
else:
    img_path = st.session_state.image_paths[st.session_state.index]
    img_name = os.path.basename(img_path)
    img = Image.open(img_path).convert('RGB').resize((149, 149))

input_array = np.expand_dims(np.array(img) / 255.0, axis=0)
start_time = time.time()
pred_lvl1 = model_lvl1.predict(input_array)[0]
pred_lvl2 = model_lvl2.predict(input_array)[0]
elapsed_time = time.time() - start_time

pred_idx1 = np.argmax(pred_lvl1)
pred_label1 = lvl1_inv.get(pred_idx1, "Unknown")
conf1 = pred_lvl1[pred_idx1]

pred_idx2 = np.argmax(pred_lvl2)
pred_label2 = lvl2_inv.get(pred_idx2, "Unknown")
conf2 = pred_lvl2[pred_idx2]

if uploaded_file:
    true_label1 = true_label2 = "unknown"
else:
    true_label2 = os.path.basename(os.path.dirname(img_path)).lower()
    true_label1 = 'abnormal' if true_label2 != 'notumor' else 'normal'

def get_result(pred, actual):
    return "TP/TN" if pred == actual else ("FP" if actual in ["normal", "notumor"] else "FN")

result_lvl1 = get_result(pred_label1, true_label1)
result_lvl2 = get_result(pred_label2, true_label2)

# Output stacked left: MetaData then Image(s)
if view_mode == "Both":
    col_meta, col_img1, col_img2 = st.columns([2, 2, 2])
else:
    col_meta, col_img1 = st.columns([2, 4])

with col_meta:
    st.markdown("### 🧾 Prediction Metadata")
    st.markdown(f"**File Name:** `{img_name}`")
    st.markdown(f"**Classification Time:** `{elapsed_time:.3f} seconds`")
    if mode in ["Level 1 Only", "Combined"]:
        st.markdown("**Level 1**")
        st.markdown(f"Prediction: `{pred_label1}` ({conf1:.2f})")
        st.markdown(f"Actual: `{true_label1}` → {result_lvl1}")
    if mode in ["Level 2 Only", "Combined"]:
        st.markdown("**Level 2**")
        st.markdown(f"Prediction: `{pred_label2}` ({conf2:.2f})")
        st.markdown(f"Actual: `{true_label2}` → {result_lvl2}")

with col_img1:
    st.image(img.resize((298, 298)), use_container_width=False)

if view_mode in ["Grad-CAM", "Both"]:
    heatmap = make_gradcam_heatmap(input_array, model_lvl2, selected_layer)
    cam_img = overlay_gradcam(img.copy(), heatmap)
    with col_img2:
        st.image(cam_img.resize((298, 298)), use_container_width=False)

st.markdown("\n")
if not uploaded_file:
    left, right = st.columns([1, 6])
    with left:
        if st.button("◀"):
            st.session_state.index = max(0, st.session_state.index - 1)
    with right:
        if st.button("▶"):
            st.session_state.index = (st.session_state.index + 1) % len(st.session_state.image_paths)
