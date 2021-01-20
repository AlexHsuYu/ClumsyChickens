import keras.layers as KL


############################################################
#  VGG16 Graph
############################################################

def vgg16_graph(input_tensor):
    """
        Build the VGG-16 graph
    """
    x = (KL.ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))(input_tensor)
    x = (KL.Conv2D(64, 3, 3, activation='relu'))(x)
    x = (KL.ZeroPadding2D((1, 1)))(x)
    x = (KL.Conv2D(64, 3, 3, activation='relu'))(x)
    C1 = x = (KL.MaxPooling2D((2, 2), strides=(2, 2)))(x)

    x = (KL.ZeroPadding2D((1, 1)))(x)
    x = (KL.Conv2D(128, 3, 3, activation='relu'))(x)
    x = (KL.ZeroPadding2D((1, 1)))(x)
    x = (KL.Conv2D(128, 3, 3, activation='relu'))(x)
    C2 = x = (KL.MaxPooling2D((2, 2), strides=(2, 2)))(x)

    x = (KL.ZeroPadding2D((1, 1)))(x)
    x = (KL.Conv2D(256, 3, 3, activation='relu'))(x)
    x = (KL.ZeroPadding2D((1, 1)))(x)
    x = (KL.Conv2D(256, 3, 3, activation='relu'))(x)
    x = (KL.ZeroPadding2D((1, 1)))(x)
    x = (KL.Conv2D(256, 3, 3, activation='relu'))(x)
    C3 = x = (KL.MaxPooling2D((2, 2), strides=(2, 2)))(x)

    x = (KL.ZeroPadding2D((1, 1)))(x)
    x = (KL.Conv2D(512, 3, 3, activation='relu'))(x)
    x = (KL.ZeroPadding2D((1, 1)))(x)
    x = (KL.Conv2D(512, 3, 3, activation='relu'))(x)
    x = (KL.ZeroPadding2D((1, 1)))(x)
    x = (KL.Conv2D(512, 3, 3, activation='relu'))(x)
    C4 = x = (KL.MaxPooling2D((2, 2), strides=(2, 2)))(x)

    x = (KL.ZeroPadding2D((1, 1)))(x)
    x = (KL.Conv2D(512, 3, 3, activation='relu'))(x)
    x = (KL.ZeroPadding2D((1, 1)))(x)
    x = (KL.Conv2D(512, 3, 3, activation='relu'))(x)
    x = (KL.ZeroPadding2D((1, 1)))(x)
    x = (KL.Conv2D(512, 3, 3, activation='relu'))(x)
    C5 = x = (KL.MaxPooling2D((2, 2), strides=(2, 2)))(x)

    return [C1, C2, C3, C4, C5]
