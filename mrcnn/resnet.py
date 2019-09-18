import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

def block(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3

    if conv_shortcut is True:
        shortcut = KL.Conv2D(4 * filters, 1, strides=stride,
                             name=name + '_0_conv')(x)
        shortcut = BatchNorm(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut, training=False)
    else:
        shortcut = x

    x = KL.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = BatchNorm(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x, training=False)
    x = KL.Activation('relu', name=name + '_1_relu')(x)

    x = KL.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv')(x)
    x = BatchNorm(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x, training=False)
    x = KL.Activation('relu', name=name + '_2_relu')(x)

    x = KL.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = BatchNorm(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x, training=False)

    x = KL.Add(name=name + '_add')([shortcut, x])
    x = KL.Activation('relu', name=name + '_out')(x)
    return x

def stack(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def ResNet(input_image, architecture):
    assert architecture in ["resnet50", "resnet101"]
    bn_axis = 3

    x = KL.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input_image)
    x = KL.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv')(x)

    x = BatchNorm(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x, training=False)
    x = KL.Activation('relu', name='conv1_relu')(x)

    x = KL.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    C1 = x = KL.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    C2 = x = stack(x, 64, 3, stride1=1, name='conv2')
    C3 = x = stack(x, 128, 4, name='conv3')
    if architecture == 'resnet50':
        C4 = x = stack(x, 256, 6, name='conv4')
    else:
        C4 = x = stack(x, 256, 23, name='conv4')
    C5 = x = stack(x, 512, 3, name='conv5')

    return [C1, C2, C3, C4, C5]

    # # Create model.
    # if True:
    #     x = KL.GlobalAveragePooling2D(name='avg_pool')(x)
    #     x = KL.Dense(1000, activation='softmax', name='probs')(x)
    #
    # model = keras.models.Model(input_image, x, name=architecture)
    # return model

def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
# if __name__ == '__main__':
#     import os
#     import cv2
#     import numpy as np
#     from keras.applications.imagenet_utils import decode_predictions
#     os.environ['CUDA_VISIBLE_DEVICES'] = '4'
#     keras.backend.tensorflow_backend.set_session(get_session())
#
#     input_image = keras.layers.Input(shape=(224, 224, 3), name="input_image")
#     model = ResNet(input_image, 'resnet101')
#     model.load_weights('/mnt/sdb/clf8113/research2/mask_RCNN-master/pretrained_models/resnet101_weights_tf_dim_ordering_tf_kernels.h5',
#                        by_name=True)
#     mean = [103.939, 116.779, 123.68]
#     x = cv2.imread('/mnt/sdb/clf8113/research2/mask_RCNN-master/test_images/20190902111334.jpg')
#     x = x[..., ::-1]
#     x = cv2.resize(x, (224, 224))
#     x = np.expand_dims(x, axis=0)
#     x = x - np.array(mean)
#     res = model.predict(x)
#     print('Predicted:', decode_predictions(res))
#     print()
#
#
