import prepare_data
import prepare_network
import numpy as np
import tensorflow as tf
batch_size = 32
epochs = 100

def train_network():
    model = prepare_network.get_network()
    x_train, y_train, x_valid, y_valid = prepare_data.get_train_valid_data()

    model.fit(np.array(x_train), np.array(y_train),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(np.array(x_valid), np.array(y_valid)),
            shuffle=True)

    #saving the  model to be used later
    fer_json = model.to_json()
    with open("fer.json", "w") as json_file:
        json_file.write(fer_json)
    model.save_weights("fer.h5")
    print("Saved model to disk")

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    tf.keras.backend.set_session(tf.Session(config=config))
    train_network()