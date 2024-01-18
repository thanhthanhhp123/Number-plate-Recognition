from modules import *


def inception():
    inp = layers.Input(shape=(hp.ss, hp.ss, 3))
    x = stem_block(inp)

    for _ in range(5):
        x = block_a(x)

    x = reduction_a(x)

    for _ in range(10):
        x = block_b(x)

    x = reduction_b(x)

    for _ in range(5):
        x = block_c(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4, activation='sigmoid')(x)
    return tf.keras.Model(inp, x)

if __name__ == '__main__':
    inception = inception(10)
    inception.summary()