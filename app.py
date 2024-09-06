from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import io
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from threading import Thread

# Paths and parameters
train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'
img_height, img_width = 224, 224
batch_size = 8

app = Flask(__name__)

def update_class_names(class_name, drink_preference, dietary_restrictions):
    try:
        if os.path.exists('class_names.json'):
            with open('class_names.json', 'r') as f:
                class_names = json.load(f)
        else:
            class_names = {}

        class_names[class_name] = {
            "drink_preference": drink_preference,
            "dietary_restrictions": dietary_restrictions
        }

        sorted_class_names = dict(sorted(class_names.items()))

        with open('class_names.json', 'w') as f:
            json.dump(sorted_class_names, f, indent=4)

        print("Updated class_names.json")
    except Exception as e:
        print(f"Error updating class_names.json: {e}")

def create_folders(class_name):
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

def augment_image(image, count=30):
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    augmented_images = []

    for _ in range(count):
        augmented_image = datagen.flow(image, batch_size=1)[0]
        augmented_image = array_to_img(augmented_image[0])
        augmented_images.append(augmented_image)

    return augmented_images

def save_images(images, class_name):
    train_folder = os.path.join(train_dir, class_name)
    valid_folder = os.path.join(valid_dir, class_name)
    test_folder = os.path.join(test_dir, class_name)

    for i, img in enumerate(images):
        if i < 3:
            img.save(os.path.join(test_folder, f'image_{i}.jpg'))
        elif i < 9:
            img.save(os.path.join(valid_folder, f'image_{i}.jpg'))
        else:
            img.save(os.path.join(train_folder, f'image_{i}.jpg'))

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def retrain_model():
    try:
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)

        num_classes = len(class_names)

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        valid_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical'
        )

        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical'
        )

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer=Adam(learning_rate=0.0002),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(
            train_generator,
            epochs=10,
            validation_data=valid_generator
        )

        model.save('staff_mobilenet_v2_model.h5')

    except Exception as e:
        print(f"Error during retraining: {e}")

def background_retrain():
    retrain_model()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        class_name = request.form.get("class_name")
        drink_preference = request.form.get("drink_preference")
        dietary_restrictions = request.form.get("dietary_restrictions")
        uploaded_file = request.files.get("file")

        if uploaded_file and class_name:
            try:
                image = Image.open(uploaded_file)
                create_folders(class_name)
                augmented_images = augment_image(image)
                save_images(augmented_images, class_name)
                update_class_names(class_name, drink_preference, dietary_restrictions)

                # Start retraining in a background thread
                thread = Thread(target=background_retrain)
                thread.start()

                # Wait for a fixed period (60 seconds) before prediction
                time.sleep(110)

                # Use the preloaded model for prediction
                model = tf.keras.models.load_model('staff_mobilenet_v2_model.h5')
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)
                predicted_class_index = np.argmax(predictions)
                predicted_class_name = list(json.load(open('class_names.json')).keys())[predicted_class_index]

                # Return the prediction as JSON
                response = jsonify({
                    "predicted_class_name": predicted_class_name,
                    "probability": f"{np.max(predictions):.2f}",
                    "drink_preference": json.load(open('class_names.json')).get(predicted_class_name, {}).get('drink_preference', 'N/A'),
                    "dietary_restrictions": json.load(open('class_names.json')).get(predicted_class_name, {}).get('dietary_restrictions', 'N/A')
                })

                return response

            except Exception as e:
                return jsonify({"error": str(e)}), 500

    return render_template("index.html")

@app.route("/real_time")
def real_time():
    return render_template("real_time.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            image = file.read()
            img_array = preprocess_image(Image.open(io.BytesIO(image)))
            model = tf.keras.models.load_model('staff_mobilenet_v2_model.h5')
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])

            # Load class names from class_names.json
            class_names = json.load(open('class_names.json'))

            # Get the predicted class name
            class_name = list(class_names.keys())[predicted_class_index]

            # Get drink preference and dietary restrictions from the JSON
            details = class_names.get(class_name, {})

            # Format the output to match the required structure
            formatted_output = {
                'Name': class_name,
                'Drink Preference': details.get('drink_preference', 'N/A'),
                'Dietary Restriction': details.get('dietary_restrictions', 'N/A')
            }

            return jsonify(formatted_output)

        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    # Dummy status check since training_complete.txt has been removed
    return jsonify({"status": "ready"})

@app.route("/edit_preferences")
def edit_preferences():
    return render_template("edit_preferences.html")

@app.route("/update_preferences", methods=["POST"])
def update_preferences():
    try:
        data = request.json
        class_name = data['class_name']
        drink_preference = data['drink_preference']
        dietary_restrictions = data['dietary_restrictions']

        # Load existing class names
        if os.path.exists('class_names.json'):
            with open('class_names.json', 'r') as f:
                class_names = json.load(f)
        else:
            return jsonify({"error": "class_names.json not found"}), 404

        # Update the specific class name
        if class_name in class_names:
            class_names[class_name]['drink_preference'] = drink_preference
            class_names[class_name]['dietary_restrictions'] = dietary_restrictions
        else:
            return jsonify({"error": "Class name not found"}), 404

        # Save the updated class names back to the JSON file
        with open('class_names.json', 'w') as f:
            json.dump(class_names, f, indent=4)

        return jsonify({"success": "Preferences updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0")
