import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras.applications import vgg19
from tensorflow.keras import layers, models

def load_and_process_img(img_path):
    img = kp_image.load_img(img_path, target_size=(256, 256))
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(img):
    img = img.squeeze()
    img = img[:, :, ::-1]
    img += 103.939
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def create_model():
    base_model = vgg19.VGG19(weights='imagenet', include_top=False, input_tensor=layers.Input(shape=(256, 256, 3)))
    base_model.trainable = False

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layers = ['block5_conv2']

    style_outputs = [base_model.get_layer(name).output for name in style_layers]
    content_outputs = [base_model.get_layer(name).output for name in content_layers]

    model = models.Model(inputs=base_model.input, outputs=style_outputs + content_outputs)
    return model

def compute_loss(model, content_image, style_image, target_image):
    content_weight = 1e4
    style_weight = 1e-2

    content_loss = tf.reduce_mean((model(content_image)[1] - model(target_image)[1]) ** 2)
    style_loss = tf.add_n([tf.reduce_mean((model(style_image)[i] - model(target_image)[i]) ** 2) for i in range(len(style_layers))])
    
    style_loss *= style_weight / len(style_layers)
    total_loss = content_weight * content_loss + style_loss
    return total_loss

@tf.function
def train_step(model, content_image, style_image, target_image, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, content_image, style_image, target_image)
    gradients = tape.gradient(loss, [target_image])
    optimizer.apply_gradients(zip(gradients, [target_image]))
    return loss

content_path = 'content.jpg'
style_path = 'style.jpg'

content_image = load_and_process_img(content_path)
style_image = load_and_process_img(style_path)
target_image = tf.Variable(content_image, dtype=tf.float32)

model = create_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

epochs = 1000
for epoch in range(epochs):
    loss = train_step(model, content_image, style_image, target_image, optimizer)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.numpy()}')

final_img = deprocess_img(target_image.numpy())
plt.imshow(final_img)
plt.axis('off')
plt.savefig('output.jpg')
plt.show()