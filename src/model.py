import multiprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

class LeNet5():
    def __init__(self, train_gen, valid_gen, step_epoch, batch_size): #TODO: add stop criterion -all images or overfit
        #TODO: add also a number of labelised images 

        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.step_epoch = step_epoch
        self.batch_size = batch_size

        #stop criterion
        self.stop_criterion = False

        #history
        self.history_val_loss = []
        self.history_val_accuracy = []
        self.history_loss = []
        self.history_accuracy = []

    def build_model(self, input_shape, optimizer, loss_funct, number_classes):

        lenet_model = Sequential()
        lenet_model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        lenet_model.add(Activation('relu'))
        lenet_model.add(Conv2D(32, (3, 3)))
        lenet_model.add(Activation('relu'))
        lenet_model.add(MaxPooling2D(pool_size=(2, 2)))
        lenet_model.add(Dropout(0.25))

        lenet_model.add(Conv2D(64, (3, 3), padding='same'))
        lenet_model.add(Activation('relu'))
        lenet_model.add(Conv2D(64, (3, 3)))
        lenet_model.add(Activation('relu'))
        lenet_model.add(MaxPooling2D(pool_size=(2, 2)))
        lenet_model.add(Dropout(0.25))

        lenet_model.add(Flatten())
        lenet_model.add(Dense(512))
        lenet_model.add(Activation('relu'))
        lenet_model.add(Dropout(0.5))
        lenet_model.add(Dense(number_classes))
        lenet_model.add(Activation('softmax'))

        lenet_model.compile(loss=loss_funct, optimizer=optimizer, metrics=['accuracy'])

        self.model = lenet_model
        
    def _step_train(self, x_train, y_train, x_validation, y_validation):
        data_size = x_train.shape[0]

        history_cbk = self.model.fit_generator(
            self.train_gen.flow(x_train, y_train, batch_size=self.batch_size),
            epochs=self.step_epoch,
            steps_per_epoch= data_size / self.step_epoch, 
            validation_data=(x_validation, y_validation),
            workers=multiprocessing.cpu_count())
        return history_cbk.history

    def _labelise(self, x_train, y_train, x_validation, y_validation):
        #finding the most "useful" data
        predictions = self.model.predict(x_validation, batch_size=self.batch_size)
        import ipdb; ipdb.set_trace()

        #labelise this data (aka. moving it from validation to train)

        return (x_train, y_train, x_validation, y_validation)

    #TODO: implement
    def _evaluate_stop(self):
        raise(NotImplementedError)

    def train(self, x_train, y_train, x_validation, y_validation):
        
        #first training
        step_history = self._step_train(x_train, y_train, x_validation, y_validation)

        #saving global history
        self.history_val_accuracy.extend(step_history['val_acc'])
        self.history_val_loss.extend(step_history['val_loss'])
        self.history_accuracy.extend(step_history['acc'])
        self.history_loss.extend(step_history['loss'])

        #labelisation of more images
        (x_train, y_train, x_validation, y_validation) = self._labelise(x_train, y_train, x_validation, y_validation)

        while(self.stop_criterion == False):
            #training
            step_history = self._step_train(x_train, y_train, x_validation, y_validation)

            #saving global history
            self.history_val_accuracy.extend(step_history['val_acc'])
            self.history_val_loss.extend(step_history['val_loss'])
            self.history_accuracy.extend(step_history['acc'])
            self.history_loss.extend(step_history['loss'])

            #labelisation of the data. If there is no data left to labelise, the program will stop
            (x_train, y_train, x_validation, y_validation) = self._labelise(x_train, y_train, x_validation, y_validation)            

            #if we are training until a certain level of accuracy (and not until there is no label left)
            self._evaluate_stop()