from Inception import inception
from params import *
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
from dataset import get_dataset
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler


model = inception()
model.summary()

model.compile(optimizer='adam', loss='mse')

X_train, X_test, y_train, y_test = get_dataset()

checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
def scheduler(epoch, lr):
    if epoch < 30:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))

lr_scheduler = LearningRateScheduler(scheduler)

tensorboard = TensorBoard(log_dir='logs')
callbacks = [checkpoint, tensorboard, lr_scheduler]

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
          validation_data=(X_test, y_test), 
          callbacks=callbacks)

model.save('inception.keras')