from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import mock

import cv2
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import monodepth2.networks as networks
from monodepth2.layers import disp_to_depth
from utilities.trained_net import TrainedNet
from monodepth2.utils import download_model_if_doesnt_exist
from monodepth2.evaluate_depth import STEREO_SCALE_FACTOR
from utilities.timer import MyTimer


def prepare_trained_net(args):
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    return TrainedNet(encoder, depth_decoder, feed_height, feed_width, device)


def test_video(args):
    """Function to predict for a single image or folder of images
    """
    timer = MyTimer()

    # FINDING INPUT VIDEO
    video = cv2.VideoCapture(args.video_path)
    print("-> Predicting on {:s} video file".format(args.video_path))

    # LOADING NET
    trained_net = prepare_trained_net(args)
    print("is cuda available:", torch.cuda.is_available())
    print("device:", trained_net.device)

    timer.start()

    # PREDICTING ON EACH IMAGE IN TURN
    success = True
    frame_counter = 0
    with torch.no_grad():
        while success:
            frame_counter += 1

            # LOAD FRAME
            success, image = video.read()

            time_loading = timer.get_time_from_last_point()

            # PREPROCESS FRAME
            original_height = image.shape[0]
            original_width = image.shape[1]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (trained_net.feed_width, trained_net.feed_height))
            input_image = transforms.ToTensor()(image).unsqueeze(0)

            time_preprocessing = timer.get_time_from_last_point()

            # PREDICTION
            input_image = input_image.to(trained_net.device)
            features = trained_net.encoder(input_image)
            outputs = trained_net.decoder(features)

            time_prediction = timer.get_time_from_last_point()

            # OUTPUT MODIFICATIONS
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

            time_output_modification = timer.get_time_from_last_point()

            # IMAGE PRESENTATION
            cv2.imshow("image", cv2.cvtColor(colormapped_im, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            time_presentation = timer.get_time_from_last_point()
            time_elapsed = time_loading + time_preprocessing + time_prediction + time_output_modification + time_presentation
            fps = 1 / time_elapsed

            print("frame: %4d ld: %7.5f pp: %7.5f pd: %7.5f om: %7.5f pr: %7.5f el: %7.5f fps: %4.1f" %
                  (frame_counter, time_loading, time_preprocessing, time_prediction, time_output_modification, time_presentation, time_elapsed, fps))

    print('-> Done!')


if __name__ == '__main__':

    input_args = mock.Mock()
    input_args.video_path = "C:\\Users\\Michal\\Videos\\EPART\\EPART_LAB2.mp4"
    input_args.model_name = "mono+stereo_640x192"
    input_args.pred_metric_depth = True
    input_args.no_cuda = False
    test_video(input_args)
