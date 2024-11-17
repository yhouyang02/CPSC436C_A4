import numpy as np
import pickle

def pickleTrainingData():
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    train_data = np.empty((0, 32*32*3))
    train_labels = []

    for i in range(1, 2):
        fileNameDataBatch = './cifar-10-batches-py/data_batch_' + str(i)
        batch = unpickle(fileNameDataBatch)
        train_data = np.vstack((train_data, batch[b'data']))
        train_labels += batch[b'labels']

    train_labels = np.array(train_labels)
    train_data = train_data.reshape(-1, 32, 32, 3) / 255.0
    
    # !!!!! NOTICE HOW THE DATA IS SAVED !!!!!!
    # Will be returned in form of:
    # train_label, train_data  = getDataBack()
    pickle.dump([train_labels,train_data], open('./train.cnn', 'wb'))

pickleTrainingData()