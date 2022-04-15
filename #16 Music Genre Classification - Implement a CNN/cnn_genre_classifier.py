import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt


DATASET_PATH = ".\#13 Music Genre Classification - Implementing a neural network\data.json"

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    # convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    
    return inputs, targets

def prepare_dataset(test_size, validation_size):
    # load data
    inputs, targets = load_data(DATASET_PATH)
    # create the train/test split
    inputs_train, inputs_test, targets_train, targets_test  = train_test_split(inputs, targets, test_size=test_size)
    # create the train/validation split
    inputs_train, inputs_validation, targets_train, targets_validation = train_test_split(inputs_train, targets_train, test_size=validation_size)
    # add third dimension
    inputs_train = inputs_train[..., np.newaxis]
    inputs_test = inputs_test[..., np.newaxis]
    inputs_validation = inputs_validation[..., np.newaxis]
    
    return inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test

def build_model(input_shape):
    # create the model
    model = keras.Sequential()
    
    # 1st convolutional layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    
    # 2nd convolutional layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    
    # 3rd convolutional layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    
    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    
    # output layer
    model.add(keras.layers.Dense(10, activation="softmax")) 
       
    return model
    
def predict(model, inputs, targets):
    inputs = inputs[np.newaxis, ...] 
    prediction = model.predict(inputs)
    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {}, Predicted index: {}".format(targets, predicted_index))
    
def plot_history(history):
    fig, axs = plt.subplots(2)
    # create the accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")    
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")    
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    
    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")    
    axs[1].plot(history.history["val_loss"], label="test error")    
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    
    plt.show()

if __name__ == "__main__":
    # create train, validation and test sets
    inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = prepare_dataset(0.25, 0.2)
    
    # build the CNN net
    input_shape = (inputs_train.shape[1], inputs_train.shape[2], inputs_train.shape[3])
    model = build_model(input_shape)
    
    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    # train the CNN 
    history = model.fit(inputs_train, targets_train, validation_data=(inputs_validation, targets_validation), batch_size=32, epochs=30)
    #model.save(".\#16 Music Genre Classification - Implement a CNN\model")
    
    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(inputs_test, targets_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    
    # make prediction on a sample
    inputs = inputs_test[100]
    targets = targets_test[100]
    predict(model, inputs, targets)
    
    # plot accuracy
    plot_history(history)