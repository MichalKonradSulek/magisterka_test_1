import torch
import mock
import os
from torchvision import transforms

import monodepth2.networks as networks
from monodepth2.layers import disp_to_depth
from monodepth2.evaluate_depth import STEREO_SCALE_FACTOR
from monodepth2.utils import download_model_if_doesnt_exist

from utilities.trained_net import TrainedNet



def _get_net_params():
    input_args = mock.Mock()
    input_args.model_name = "mono+stereo_640x192"
    input_args.pred_metric_depth = True
    input_args.no_cuda = False
    return input_args


def _prepare_trained_net(args):
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


class Monodepth2Runner:
    def __init__(self):
        args = _get_net_params()
        self.net = _prepare_trained_net(args)
        self.frame_shape = (192, 640)

    def generate_depth(self, resized_rgb_image):
        with torch.no_grad():
            # PREPROCESS FRAME
            input_image = transforms.ToTensor()(resized_rgb_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(self.net.device)
            features = self.net.encoder(input_image)
            outputs = self.net.decoder(features)

            # OUTPUT MODIFICATIONS
            disp = outputs[("disp", 0)]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
        return metric_depth
