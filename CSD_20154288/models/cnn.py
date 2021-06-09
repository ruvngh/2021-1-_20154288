from tensorflow.keras.layers import Activation, Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2


def mini_XCEPTION(input_shape, num_classes, num_regs, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4 for exp.
    # exp branch.
    '''
    last repeated module separated by ext, etc branch.
    '''
    residual_exp = Conv2D(128, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
    residual_exp = BatchNormalization()(residual_exp)

    x_exp = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
    x_exp = BatchNormalization()(x_exp)
    x_exp = Activation('relu')(x_exp)
    x_exp = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x_exp)
    x_exp = BatchNormalization()(x_exp)

    x_exp = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x_exp)
    x_exp = layers.add([x_exp, residual_exp])

    x_exp = Conv2D(num_classes, (3, 3),
                   kernel_regularizer=regularization,
                   padding='same')(x_exp)
    x_exp = GlobalAveragePooling2D()(x_exp)

    # module 4 for etc branch.
    residual_etc = Conv2D(128, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
    residual_etc = BatchNormalization()(residual_etc)

    x_etc = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
    x_etc = BatchNormalization()(x_etc)
    x_etc = Activation('relu')(x_etc)
    x_etc = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x_etc)
    x_etc = BatchNormalization()(x_etc)

    x_etc = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x_etc)
    x_etc = layers.add([x_etc, residual_etc])

    # val, aro branch.
    x_etc = Conv2D(num_regs, (3, 3),
                   kernel_regularizer=regularization,
                   padding='same')(x_etc)
    x_etc = GlobalAveragePooling2D()(x_etc)

    output_exp = Activation('softmax', name='predictions_exp')(x_exp)
    output_etc = Activation('linear', name='predictions_etc')(x_etc)

    model = Model(img_input, outputs=[output_exp, output_etc])
    return model
