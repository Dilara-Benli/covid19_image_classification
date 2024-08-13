import matplotlib.pyplot as plt
import tensorflow as tf
import json
# UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
# The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

class ModelTraining:
    def __init__(self, epochs, learning_rate): 
        self.TRAINING_DATA_DIR = 'Covid19-dataset/train'
        self.VALID_DATA_DIR = 'Covid19-dataset/val'
        self.IMAGE_SHAPE = (640, 640)
        self.BATCH_SIZE = 32
        self.num_classes = 3
        self.EPOCHS = epochs
        self.learning_rate = learning_rate

        #self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        # data augmentation
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, 
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
            )
        self.train_generator = self.datagen.flow_from_directory(self.TRAINING_DATA_DIR, shuffle=True, target_size=self.IMAGE_SHAPE)
        self.valid_generator = self.datagen.flow_from_directory(self.VALID_DATA_DIR, shuffle=False, target_size=self.IMAGE_SHAPE)

    def build_model(self):
        model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=(640, 640, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'), 
        tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def create_model(self):
        model = self.build_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])
        
        history = model.fit(self.train_generator,
                    steps_per_epoch=self.train_generator.samples // self.BATCH_SIZE,
                    epochs=self.EPOCHS,
                    validation_data=self.valid_generator,
                    validation_steps=self.valid_generator.samples // self.BATCH_SIZE,
                    verbose=1
                    )
        
        return model, history
    
    def save_model_and_history(self, model, history, model_path, history_path):
        model.save(model_path)
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
        print(f"Model saved to {model_path} and history saved to {history_path}")

    def load_model(self, model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    
    def load_history(self, history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"History loaded from {history_path}")
        return history
    
    def plot_evaluation_metrics(self, history, metrics):
        num_metrics = len(metrics)
        num_cols = 2  # sütun
        num_rows = (num_metrics + num_cols - 1) // num_cols  # satır

        plt.figure(figsize=(3 * num_cols, 3 * num_rows))

        for i, metric in enumerate(metrics):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.plot(history[metric], label='Train ' + metric.capitalize()) 
            plt.plot(history[f'val_{metric}'], label='Val ' + metric.capitalize())
            #plt.title(f'Train and Val {metric.capitalize()}') # Train and Validation
            plt.title(metric)
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()

        plt.tight_layout()
        plt.show()