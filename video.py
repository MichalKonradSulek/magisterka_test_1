from __future__ import absolute_import, division, print_function

import os

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import mock
import numpy as np
import torch
from torchvision import transforms

import monodepth2.networks as networks
from monodepth2.utils import download_model_if_doesnt_exist
from monodepth2.layers import disp_to_depth
from monodepth2.evaluate_depth import STEREO_SCALE_FACTOR
from utilities.timer import MyTimer
from utilities.trained_net import TrainedNet


class Monodepth2VideoInterpreter:
    def __init__(self, video_path):
        args = Monodepth2VideoInterpreter.__get_net_params()
        self.net = Monodepth2VideoInterpreter.__prepare_trained_net(args)
        self.video = Monodepth2VideoInterpreter.__load_video(video_path)
        self.timer = MyTimer()
        self.frame_counter = 0
        self.show_original = True
        self.print_times = True
        self.frame_shape = (192, 640)

    @staticmethod
    def __get_net_params():
        input_args = mock.Mock()
        input_args.model_name = "mono+stereo_640x192"
        input_args.pred_metric_depth = True
        input_args.no_cuda = False
        return input_args

    @staticmethod
    def __prepare_trained_net(args):
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

    @staticmethod
    def __load_video(path):
        video = cv2.VideoCapture(path)
        print("-> Predicting on {:s} video file".format(path))
        return video

    def get_next_depth_frame(self):
        self.frame_counter += 1
        self.timer.start()

        # LOAD FRAME
        success, image = self.video.read()

        if success:
            time_loading = self.timer.get_time_from_last_point()

            # SHOW ORIGINAL IF NEEDED
            if self.show_original:
                cv2.imshow("original", image)

            with torch.no_grad():
                # PREPROCESS FRAME
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.net.feed_width, self.net.feed_height))
                input_image = transforms.ToTensor()(image).unsqueeze(0)

                time_preprocessing = self.timer.get_time_from_last_point()

                # PREDICTION
                input_image = input_image.to(self.net.device)
                features = self.net.encoder(input_image)
                outputs = self.net.decoder(features)

                time_prediction = self.timer.get_time_from_last_point()

                # OUTPUT MODIFICATIONS
                disp = outputs[("disp", 0)]
                scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()

                time_output_modification = self.timer.get_time_from_last_point()

            # # IMAGE PRESENTATION
            # cv2.imshow("image", cv2.cvtColor(colormapped_im, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(1)
            # time_presentation = timer.get_time_from_last_point()

            if self.print_times:
                time_elapsed = time_loading + time_preprocessing + time_prediction + time_output_modification
                fps = 1 / time_elapsed
                print("frame: %4d ld: %7.5f pp: %7.5f pd: %7.5f om: %7.5f el: %7.5f fps: %4.1f" %
                      (self.frame_counter, time_loading, time_preprocessing, time_prediction, time_output_modification,
                       time_elapsed, fps))

            return success, metric_depth.squeeze()
        else:
            return success, None

    def get_next_disparity_frame(self):
        self.frame_counter += 1
        self.timer.start()

        # LOAD FRAME
        success, image = self.video.read()

        if success:
            time_loading = self.timer.get_time_from_last_point()

            # SHOW ORIGINAL IF NEEDED
            if self.show_original:
                cv2.imshow("original", image)

            with torch.no_grad():
                # PREPROCESS FRAME
                original_height = image.shape[0]
                original_width = image.shape[1]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.net.feed_width, self.net.feed_height))
                input_image = transforms.ToTensor()(image).unsqueeze(0)

                time_preprocessing = self.timer.get_time_from_last_point()

                # PREDICTION
                input_image = input_image.to(self.net.device)
                features = self.net.encoder(input_image)
                outputs = self.net.decoder(features)

                time_prediction = self.timer.get_time_from_last_point()

                # OUTPUT MODIFICATIONS
                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

                time_output_modification = self.timer.get_time_from_last_point()

            # # IMAGE PRESENTATION
            # cv2.imshow("image", cv2.cvtColor(colormapped_im, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(1)
            # time_presentation = timer.get_time_from_last_point()

            if self.print_times:
                time_elapsed = time_loading + time_preprocessing + time_prediction + time_output_modification
                fps = 1 / time_elapsed
                print("frame: %4d ld: %7.5f pp: %7.5f pd: %7.5f om: %7.5f el: %7.5f fps: %4.1f" %
                      (self.frame_counter, time_loading, time_preprocessing, time_prediction, time_output_modification,
                       time_elapsed, fps))

            return success, colormapped_im
        else:
            return success, None


def test_video(video_path):
    video_provider = Monodepth2VideoInterpreter(video_path)
    success, disparity_frame = video_provider.get_next_disparity_frame()
    while success:
        cv2.imshow("image", cv2.cvtColor(disparity_frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        success, disparity_frame = video_provider.get_next_disparity_frame()


if __name__ == '__main__':
    video_path = "C:\\Users\\Michal\\Videos\\VID_20220411_212615471.mp4"
    test_video(video_path)
