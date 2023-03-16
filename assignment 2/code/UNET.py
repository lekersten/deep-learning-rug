import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, ResNet50


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)

    return x

def build_mobilenetv2_unet(input_shape):
    res = input_shape[0]
    res_size = [res/1, res/2, res/4, res/8]
    inputs = Input(shape=input_shape)

    encoder = MobileNetV2(include_top=False,
        input_tensor=inputs, alpha=1.4)


    s1 = encoder.get_layer("input_1").output                
    s2 = encoder.get_layer("block_1_expand_relu").output    
    s3 = encoder.get_layer("block_3_expand_relu").output   
    s4 = encoder.get_layer("block_6_expand_relu").output   

    b1 = encoder.get_layer("block_13_expand_relu").output  

    d1 = decoder_block(b1, s4, res_size[0])                      
    d2 = decoder_block(d1, s3, res_size[1])                     
    d3 = decoder_block(d2, s2, res_size[2])                       
    d4 = decoder_block(d3, s1, res_size[3])

    outputs = Conv2D(3, 3, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="MobileNetV2_U-Net")
    return model

def build_resnet50_unet(input_shape):
    res = input_shape[0]
    res_size = [res/1, res/2, res/4, res/8]
    inputs = Input(input_shape)

    resnet50 = ResNet50(include_top=False, input_tensor=inputs)

    s1 = resnet50.get_layer("input_1").output           
    s2 = resnet50.get_layer("conv1_relu").output        
    s3 = resnet50.get_layer("conv2_block3_out").output  
    s4 = resnet50.get_layer("conv3_block4_out").output  

    b1 = resnet50.get_layer("conv4_block6_out").output  

    d1 = decoder_block(b1, s4, res_size[0])                      
    d2 = decoder_block(d1, s3, res_size[1])                     
    d3 = decoder_block(d2, s2, res_size[2])                       
    d4 = decoder_block(d3, s1, res_size[3])

    outputs = Conv2D(3, 3, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="ResNet50_U-Net")
    return model
