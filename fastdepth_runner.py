import os
import torch
from torchvision import transforms

from utilities import fast_depth_custom_pickle


def _prepare_trained_net(device):
    model_path = "models/mobilenet-nnconv5dw-skipadd-pruned.pth.tar"

    assert os.path.isfile(model_path), \
        "=> no model found at '{}'".format(model_path)
    print("=> loading model '{}'".format(model_path))
    checkpoint = torch.load(model_path, pickle_module=fast_depth_custom_pickle)
    if type(checkpoint) is dict:
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
    else:
        model = checkpoint
    model.to(device)
    model.eval()
    return model


class FastDepthRunner:
    def __init__(self, on_cuda=True):
        self.frame_shape = (224, 224)
        self.device = "cuda" if on_cuda else "cpu"
        self.net = _prepare_trained_net(self.device)

    def generate_depth(self, resized_rgb_image):
        # PREPROCESS FRAME
        input_tensor = transforms.ToTensor()(resized_rgb_image / 255).float().unsqueeze(0).to(self.device)

        # PREDICTION
        with torch.no_grad():
            net_output = self.net(input_tensor)

        # OUTPUT MODIFICATIONS
        metric_depth = net_output.cpu().detach().numpy().squeeze()

        return metric_depth
