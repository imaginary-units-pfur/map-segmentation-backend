import numpy as np
import cv2
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CFG:
    model_name = "Unet"
    backbone = "timm-efficientnet-b7"
    ckpt_path = "./best_epoch_pretrained_case_data.bin"
    img_size = [1024, 1024]
    num_classes = 1
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda else "cpu")


def build_model(backbone, num_classes, device):
    model = smp.Unet(
        encoder_name=backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.) classes=num_classes,  # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(device)
    return model


def load_model(backbone, num_classes, device, path):
    model = build_model(backbone, num_classes, device)
    model.load_state_dict(torch.load(path, map_location=CFG.device))
    model.eval()
    return model


def load_img(path, size):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, size)
    img = img.astype("float32")  # original is uint16
    img = img / 255
    # img = A.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img


def img_from_bytes(byte_img, size):
    img = np.fromstring(byte_img, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, size)
    img = img.astype("float32")  # original is uint16
    img = img / 255
    # img = A.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img

model = load_model(CFG.backbone, CFG.num_classes, CFG.device, CFG.ckpt_path)


threshold = 0.5


def segment(byte_img):
    # image = load_img(
    #     "./train/images/train_image_015.png",
    #     (1024, 1024),
    # )
    image = img_from_bytes(byte_img, (1024, 1024))
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image = image.cuda() if CFG.is_cuda else image.cpu()
    with torch.no_grad():
        pred = model(image)
        pred = (nn.Sigmoid()(pred) >= threshold).double()
        pred = pred.cpu().numpy().astype(np.uint8)
        return pred[0, 0, :, :] * 255

        # print("preparing plog")
        # prepare_plot(
        #     image, load_img("./train/masks/train_mask_014.png", (1024, 1024)), pred
        # )
