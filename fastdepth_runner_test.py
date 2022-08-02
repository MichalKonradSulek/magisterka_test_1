import cv2

from fastdepth_runner import FastDepthRunner


def fit_image_for_net(img):
    if img.shape[0] > img.shape[1]:
        half_difference = int((img.shape[0] - img.shape[1]) / 2)
        cropped_image = img[half_difference:(half_difference + img.shape[1]), :]
    elif img.shape[1] > img.shape[0]:
        half_difference = int((img.shape[1] - img.shape[0]) / 2)
        cropped_image = img[:, half_difference:(half_difference + img.shape[0])]
    else:
        cropped_image = img
    return cv2.resize(cropped_image, (224, 224))


if __name__ == "__main__":
    model_path = "models/mobilenet-nnconv5dw-skipadd-pruned.pth.tar"
    image_path = "C:\\Users\\Michal\\Pictures\\Test\\test9.jpg"

    fast_depth_runner = FastDepthRunner(on_cuda=False)

    input_image = cv2.imread(image_path)
    resized_input = fit_image_for_net(input_image)
    switched_colors = cv2.cvtColor(resized_input, cv2.COLOR_BGR2RGB)

    output = fast_depth_runner.generate_depth(switched_colors)

    # cv2.imshow("original", input_image)
    cv2.imshow("resized", resized_input)
    cv2.imshow("depth", output / 10)
    cv2.waitKey(0)
