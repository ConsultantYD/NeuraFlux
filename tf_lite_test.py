import numpy as np
import tensorflow as tf
import time
from shutil import rmtree


class DQNEstimator:
    def __init__(self):
        self.model = self._build_model()

    @staticmethod
    def _build_model():
        # Build model with two outputs
        input_layer = tf.keras.layers.Input(shape=(28, 28), name="input")
        x = tf.keras.layers.LSTM(20, return_sequences=True)(input_layer)
        x = tf.keras.layers.Flatten()(x)
        output1 = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="output1")(x)
        output2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="output2")(x)
        output3 = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="output3")(x)

        model = tf.keras.Model(
            inputs=[input_layer], outputs=[output1, output2, output3]
        )

        model.compile(
            optimizer="adam",
            loss=[
                "sparse_categorical_crossentropy",
                "sparse_categorical_crossentropy",
                "sparse_categorical_crossentropy",
            ],
            metrics=["accuracy"],
        )
        return model

    def predict(self, x):
        return self.model.predict(x)

    def lite_predict(self, x):
        # Run the model with TensorFlow Lite
        interpreter = tf.lite.Interpreter(model_content=self.lite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], x[0:1, :, :])
        interpreter.invoke()

        result = [interpreter.get_tensor(output_details[i]["index"]) for i in range(3)]

        # Please note: TfLite fused Lstm kernel is stateful, so we need to reset
        # the states.
        # Clean up internal states.
        interpreter.reset_all_variables()
        return result

    def train(self, x, y, epochs):
        # Double y data to simulate double outputs
        new_y = [y, y, y]
        self.model.fit(x, new_y, epochs=epochs)

    def convert_to_tflite(self):
        run_model = tf.function(lambda x: self.model(x))

        # Fix the input size.
        BATCH_SIZE = 1
        STEPS = 28
        INPUT_SIZE = 28
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], self.model.inputs[0].dtype)
        )

        # model directory.
        MODEL_DIR = "keras_lstm"
        self.model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

        converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        self.lite_model = converter.convert()

        # Clean up the model directory.
        rmtree(MODEL_DIR)


# Initialize the model
estimator = DQNEstimator()

# Data
# Load MNIST dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Train the model
estimator.train(x_train[:1000], y_train[:1000], epochs=1)

estimator.convert_to_tflite()

# Run the model with TensorFlow to get expected results.
TEST_CASES = 10

for i in range(TEST_CASES):
    x_in = x_test[i : i + 1]

    start = time.time()
    expected = estimator.predict(x_in)
    print(expected)
    print()
    start = time.time()
    lite_result = estimator.lite_predict(x_in)
    print(lite_result)
    print("~~~~~~~~~~~")
