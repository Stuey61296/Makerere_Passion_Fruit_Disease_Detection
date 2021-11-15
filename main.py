# imports
import warnings

from sklearn import model_selection

from Data import Data
# Classes
from Networks.CNN import CNN_Network
from Utils import avg_std, clean_data_for_labels, convert_to_array

warnings.filterwarnings("ignore")


def train_network(network, epoch, batch_size, train_data, train_labels, validation_data, validation_labels,
                  learning_rate):
    return network.train(train_x=train_data, train_y=train_labels, validation_x=validation_data,
                         validation_y=validation_labels, epoch=epoch, batch_size=batch_size,
                         learning_rate=learning_rate)


def main():
    # Load the dataset
    data = Data(base='Data', file='Train.csv')
    dataset = data.get_data_frame()
    train_arr_cnn = []
    val_arr_cnn = []
    test_arr_cnn = []

    # Parameters
    epoch = 350
    batch_size = 64
    loop_range = 2
    test_split = 0.2
    val_split = 0.15
    learning_rate = 0.001

    for i in range(loop_range):
        # Split the data
        train_data, test_data = model_selection.train_test_split(dataset, test_size=test_split)
        train_data, validation_data = model_selection.train_test_split(train_data, test_size=val_split)

        train_data, train_labels = clean_data_for_labels(data=train_data)

        validation_data, validation_labels = clean_data_for_labels(data=validation_data)

        test_data, test_labels = clean_data_for_labels(data=test_data)

        trained_cnn = train_network(network=CNN_Network(input_shape=(200, 200, 3)), epoch=epoch,
                                    batch_size=batch_size, train_data=train_data['Image_ID'], train_labels=train_labels,
                                    validation_data=validation_data['Image_ID'], validation_labels=validation_labels,
                                    learning_rate=learning_rate)

        test_arr_cnn.append(trained_cnn.test(data=convert_to_array(test_data['Image_ID'], "Train"), labels=test_labels,
                                             batch_size=batch_size))
        train_arr_cnn.append(trained_cnn.test(data=convert_to_array(train_data['Image_ID'], "Train"),
                                              labels=train_labels, batch_size=batch_size))
        
        val_arr_cnn.append(trained_cnn.test(data=convert_to_array(validation_data['Image_ID'], "Train"),
                                            labels=validation_labels, batch_size=batch_size))

    print()
    avg_std(data=train_arr_cnn, name='Training')
    avg_std(data=val_arr_cnn, name='Validation')
    avg_std(data=test_arr_cnn, name='Testing')
    return


if __name__ == "__main__":
    main()
