import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# 1. Load the model 
model_path = 'pneumonia_best_model.h5' 
if not os.path.exists(model_path):
    model_path = 'pneumonia_classifier.h5'

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# 2. Find the index and name of the last convolutional layer
last_conv_idx = -1
last_conv_layer_name = ""
for idx, layer in enumerate(model.layers):
    if 'conv' in layer.name.lower():
        last_conv_idx = idx
        last_conv_layer_name = layer.name

print(f"Using Layer: {last_conv_layer_name} at index {last_conv_idx}")

# 3. Dynamic Grad-CAM Function (No Model Sub-classing!)
def make_gradcam_heatmap(img_array, model, last_conv_idx):
    with tf.GradientTape() as tape:
        x = img_array
        last_conv_layer_output = None
        
        # Pass the input manually through every single layer
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if i == last_conv_idx:
                last_conv_layer_output = x
                # Tell the tape to actively watch this layer's values
                tape.watch(last_conv_layer_output)
        
        preds = x
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute the gradients of the top predicted class w.r.t the feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

# --- RUN VISUALIZATION ---

img_path = r'C:\Users\User\Desktop\Work\Deep_Learning_Projects\Pneumonia_Classifier_Using_ChestXRay\dataset\test\PNEUMONIA\person46_virus_96.jpeg'

if not os.path.exists(img_path):
    print(f"ERROR: Cannot find image at {img_path}")
else:
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Generate Heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_idx)
    
    # Process visual formats
    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.width, img.height))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose
    superimposed_img = jet_heatmap * 0.4 + tf.keras.preprocessing.image.img_to_array(img)
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    # Display Results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original X-Ray")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM Heatmap (AI Focus)")
    plt.axis('off')
    
    plt.savefig('gradcam_output.png')
    print("Output successfully saved as 'gradcam_output.png'")
    plt.show()
