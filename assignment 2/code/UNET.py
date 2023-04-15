from tensorflow.keras.layers import Conv2D, SeparableConv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Input
from tensorflow.keras.applications import MobileNetV2, ResNet50

def conv_block(input, num_filters):
    x = SeparableConv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = SeparableConv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def decoder_block(input, skip_features, num_filters):
    x = UpSampling2D((2, 2))(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_mobilenetv2_unet(input_shape=(32, 32, 1)):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained MobileNetV2 Model """
    mobilenetv2 = MobileNetV2(include_top=False, weights=None, input_tensor=inputs, alpha=1.0)

    s1 = mobilenetv2.get_layer("expanded_conv_project_BN").output      # (64 x 64)
    s2 = mobilenetv2.get_layer("block_3_expand_relu").output           # (32 x 32)
    s3 = mobilenetv2.get_layer("block_6_expand_relu").output           # (16 x 16)
    s4 = mobilenetv2.get_layer("block_13_expand_relu").output          # (8 x 8)

    """ Bridge """
    b1 = mobilenetv2.get_layer("out_relu").output                      # (4 x 4)

    """ Decoder """
    d1 = decoder_block(b1, s4, 1024)  # (8 x 8)
    d2 = decoder_block(d1, s3, 512)   # (16 x 16)
    d3 = decoder_block(d2, s2, 256)   # (32 x 32)
    d4 = decoder_block(d3, s1, 128)   # (64 x 64)

    """ Output """
    outputs = Conv2D(2, 1, padding="same", activation="tanh")(d4)
    outputs = UpSampling2D((2, 2))(outputs)

    model = Model(inputs, outputs, name="MobileNetV2_U-Net")
    return model


def build_resnet50_unet(input_shape=(32,32,1)):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights=None, input_tensor=inputs)

    s1 = resnet50.get_layer("conv1_relu").output        # (64 x 64)
    s2 = resnet50.get_layer("conv2_block3_out").output  # (32 x 32)
    s3 = resnet50.get_layer("conv3_block4_out").output  # (16 x 16)
    s4 = resnet50.get_layer("conv4_block6_out").output  # (8 x 8)

    """ Bridge """
    b1 = resnet50.get_layer("conv5_block3_out").output  # (4 x 4)

    """ Decoder """
    d1 = decoder_block(b1, s4, 1024)  # (8 x 8)
    d2 = decoder_block(d1, s3, 512)   # (16 x 16)
    d3 = decoder_block(d2, s2, 256)   # (32 x 32)
    d4 = decoder_block(d3, s1, 128)   # (64 x 64)


    """ Output """
    outputs = Conv2D(2, 1, padding="same", activation="tanh")(d4)
    outputs = UpSampling2D((2, 2))(outputs)

    model = Model(inputs, outputs, name="ResNet50_U-Net")
    return model
