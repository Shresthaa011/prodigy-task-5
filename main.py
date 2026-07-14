import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

print("=" * 60)
print("TASK-05 : NEURAL STYLE TRANSFER")
print("=" * 60)

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load Style Transfer Model
print("Loading model...")
model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# Function to load image
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((512, 512))
    img = np.array(img).astype(np.float32) / 255.0
    img = tf.convert_to_tensor(img)
    img = img[tf.newaxis, :]
    return img

print("Loading images...")

# Make sure these files are uploaded in Colab
content_image = load_image("content.jpg")
style_image = load_image("style.jpg")

print("Applying Style Transfer...")

stylized_image = model(content_image, style_image)[0]

# Save image
output_path = "outputs/stylized_image.png"

plt.imshow(tf.squeeze(stylized_image))
plt.axis("off")
plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
plt.show()

print("\n✅ Style Transfer Completed Successfully!")
print("Image saved as:", output_path)
