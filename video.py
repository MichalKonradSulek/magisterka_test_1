from __future__ import absolute_import, division, print_function
import cv2


def _load_video(path):
    video = cv2.VideoCapture(path)
    print("-> Predicting on {:s} video file".format(path))
    return video


class Monodepth2VideoInterpreter:
    def __init__(self, video_path, depth_generator, show_original=True):
        self.depth_generator = depth_generator
        self.frame_shape = depth_generator.frame_shape
        self.video = _load_video(video_path)
        self.frame_counter = 0
        self.show_original = show_original

    def get_next_depth_frame(self):
        self.frame_counter += 1

        # LOAD FRAME
        success, image = self.video.read()

        if success:
            # SHOW ORIGINAL IF NEEDED
            if self.show_original:
                cv2.imshow("original", image)

            # PREPROCESS FRAME
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.frame_shape[1], self.frame_shape[0]))

            # GENERATE DEPTH
            metric_depth = self.depth_generator.generate_depth(image)

            return success, metric_depth.squeeze()
        else:
            return success, None


# def test_video(video_path):
#     video_provider = Monodepth2VideoInterpreter(video_path)
#     success, disparity_frame = video_provider.get_next_disparity_frame()
#     while success:
#         cv2.imshow("image", cv2.cvtColor(disparity_frame, cv2.COLOR_RGB2BGR))
#         cv2.waitKey(1)
#         success, disparity_frame = video_provider.get_next_disparity_frame()
#
#
# if __name__ == '__main__':
#     video_path = "C:\\Users\\Michal\\Videos\\VID_20220411_212615471.mp4"
#     test_video(video_path)
