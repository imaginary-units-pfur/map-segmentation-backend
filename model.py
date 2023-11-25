import numpy as np
import cv2
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CFG:
    model_name = "Unet"
    backbone = "efficientnet-b7"
    ckpt_path = "./best_epoch.bin"
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


def load_img(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    init_shape = img.shape[:2]
    if size:
        img = cv2.resize(img, size)
    img = img.astype("float32")  # original is uint16
    img = img / 255
    # img = A.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img, init_shape


def img_from_bytes(byte_img, size=None):
    img = np.fromstring(byte_img, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    init_shape = img.shape[:2]
    if size:
        img = cv2.resize(img, size)
    img = img.astype("float32")  # original is uint16
    img = img / 255
    # img = A.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img, init_shape


model = load_model(CFG.backbone, CFG.num_classes, CFG.device, CFG.ckpt_path)


threshold = 0.5


def prediction_patched(model, image, patch_size, step):
    #     image = torch.nn.functional.pad(image, (step, step, step, step))
    _, input_h, input_w = image.shape

    segm_img = torch.zeros((input_h, input_w), dtype=torch.float32)
    patch_num = 1
    for i in range(0, input_h, step):  # Steps of 256
        for j in range(0, input_w, step):  # Steps of 256
            input_image = torch.zeros((3, patch_size, patch_size))

            single_patch = image[:, i : i + patch_size, j : j + patch_size]
            #             single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
            single_patch_shape = single_patch.shape[-2:]
            #             single_patch_input = np.expand_dims(single_patch, 0)
            #             print(single_patch_input.shape)

            input_image[
                :, : single_patch_shape[0], : single_patch_shape[1]
            ] = single_patch

            with torch.no_grad():
                input_image = input_image.cuda() if CFG.is_cuda else image.cpu()
                single_patch_prediction = model(input_image.unsqueeze(0))
                #                 single_patch_prediction = nn.Sigmoid()(single_patch_prediction)
                single_patch_prediction = single_patch_prediction.cpu().numpy()

            result_image = single_patch_prediction[
                :, :, : single_patch_shape[0], : single_patch_shape[1]
            ]

            #             segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
            #             single_patch_prediction = np.expand_dims(np.expand_dims(single_patch_prediction, 0), 0)
            segm_img[
                i : i + single_patch_shape[0], j : j + single_patch_shape[1]
            ] += result_image

            patch_num += 1
    #     return segm_img.numpy()[step:-step, step:-step]
    return segm_img


import ttach as tta

transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ]
)


def do_tta(image):
    masks = []
    for (
        transformer
    ) in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
        # augment image
        image = image.cuda() if CFG.is_cuda else image.cpu()
        augmented_image = transformer.augment_image(image)

        # pass to model
        with torch.no_grad():
            model_output = model(augmented_image).detach().cpu()

        # reverse augmentation for mask and label
        deaug_mask = transformer.deaugment_mask(model_output)

        # save results
        masks.append(deaug_mask)

    masks = torch.stack(masks)[:, 0, 0].sum(0)
    return masks


def prediction_patched_tta(model, image, patch_size, step):
    #     image = torch.nn.functional.pad(image, (step, step, step, step))
    _, input_h, input_w = image.shape

    segm_img = torch.zeros((input_h, input_w), dtype=torch.float32)
    patch_num = 1
    for i in range(0, input_h, step):  # Steps of 256
        for j in range(0, input_w, step):  # Steps of 256
            input_image = torch.zeros((3, patch_size, patch_size))

            single_patch = image[:, i : i + patch_size, j : j + patch_size]
            #             single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
            single_patch_shape = single_patch.shape[-2:]
            #             single_patch_input = np.expand_dims(single_patch, 0)
            #             print(single_patch_input.shape)

            input_image[
                :, : single_patch_shape[0], : single_patch_shape[1]
            ] = single_patch
            single_patch_prediction = do_tta(input_image.unsqueeze(0))
            #             with torch.no_grad():
            #                 single_patch_prediction = model(input_image.cuda().unsqueeze(0))
            single_patch_prediction = (
                single_patch_prediction.cpu().unsqueeze(0).unsqueeze(0).numpy()
            )

            result_image = single_patch_prediction[
                :, :, : single_patch_shape[0], : single_patch_shape[1]
            ]

            #             segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
            #             single_patch_prediction = np.expand_dims(np.expand_dims(single_patch_prediction, 0), 0)
            segm_img[
                i : i + single_patch_shape[0], j : j + single_patch_shape[1]
            ] += result_image

            patch_num += 1
    #     return segm_img.numpy()[step:-step, step:-step]
    return segm_img


def predict_patched(byte_img, step, use_tta=False):
    image = torch.from_numpy(img_from_bytes(byte_img, None)[0]).permute(2, 0, 1)
    if use_tta:
        mask_patched = prediction_patched_tta(model, image, patch_size=1024, step=step)
    else:
        mask_patched = prediction_patched(model, image, patch_size=1024, step=step)
    return mask_patched


def segment(byte_img):
    # image = load_img(
    #     "./train/images/train_image_015.png",
    #     (1024, 1024),
    # )
    # image = img_from_bytes(byte_img, (1024, 1024))
    # image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    # image = image.cuda() if CFG.is_cuda else image.cpu()
    with torch.no_grad():
        pred_patched = predict_patched(byte_img, step=256, use_tta=False)
        pred = (nn.Sigmoid()(pred_patched) > 0.5).numpy().astype(np.uint8)
        pred = pred.cpu().numpy().astype(np.uint8)
        return pred[0, 0, :, :] * 255

        # print("preparing plog")
        # prepare_plot(
        #     image, load_img("./train/masks/train_mask_014.png", (1024, 1024)), pred
        # )
