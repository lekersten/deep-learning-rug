import dataloader
import UNET
import utils
import sys

if __name__ == "__main__":
    train_dataset, test_dataset, A_weights, B_weights = dataloader.load_data(255)

    input_shape = (32, 32, 1)

    if sys.argv[1] == "resnet":
        model = UNET.build_resnet50_unet(input_shape)
    elif sys.argv[1] == "mobilenet":
        model = UNET.build_mobilenetv2_unet(input_shape)

    if sys.argv[2] == "ssim":
        loss = utils.WeightedSSIMLoss(A_weights, B_weights)
    elif sys.argv[2] == "mae":
        loss = utils.WeightedMeanAbsoluteError(A_weights, B_weights)

    utils.kfold_cv(model, train_dataset, loss, 10, sys.argv[1], sys.argv[2])
