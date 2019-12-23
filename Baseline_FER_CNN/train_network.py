import prepare_data
import prepare_network

def train_network():
    x_train, y_train, x_test, y_test = prepare_data.get_train_test_data()
    data_generator, model, callbacks = prepare_network.get_network()
    batch_size = 32
    num_epochs = 110
    history = model.fit_generator(data_generator.flow(x_train, y_train, batch_size),
                        steps_per_epoch = len(x_train) / batch_size,
                        epochs = num_epochs, verbose = 1, callbacks = callbacks,
                        validation_data = (x_test, y_test))
    model.save_weights("model.h5")
    print("Model saved")

train_network()