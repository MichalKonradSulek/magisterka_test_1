import cv2


class ImageWindowController:
    def __init__(self, window_name="window", callback_function=None, wait_for=None, wait_keys_dict=None,
                 run_cv_waitkey=True, resize_factor_x_y=None):
        self.window_name = window_name
        self.callback_function = callback_function
        self.wait_for = wait_for
        self.wait_keys_dict = wait_keys_dict
        self.run_cv_waitkey = run_cv_waitkey
        self.resize_factor_x_y = resize_factor_x_y

        self._create_window()

        self.is_waiting_for_key = False

    def _create_window(self):
        cv2.namedWindow(self.window_name)
        if self.callback_function is not None:
            cv2.setMouseCallback(self.window_name, lambda event, x, y, flags, param: self._window_callback(event, x, y))

    def _window_callback(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.callback_function(x, y)

    def stop_waiting_for_key(self):
        self.is_waiting_for_key = False

    def _wait_for_key(self):
        if self.wait_for is not None:
            cv2.waitKey(self.wait_for)
        elif self.wait_keys_dict is not None:
            self.is_waiting_for_key = True
            while self.is_waiting_for_key:
                key_pressed = cv2.waitKey(100)
                key_number = key_pressed & 0xFF
                if key_number in self.wait_keys_dict:
                    self.wait_keys_dict[key_number]()
        else:
            cv2.waitKey(0)

    def show_image(self, image, do_not_run_waitkey=False):
        if self.resize_factor_x_y is not None:
            target_size = (image.shape[1] * self.resize_factor_x_y[0], image.shape[0] * self.resize_factor_x_y[1])
            image_to_show = cv2.resize(image, target_size)
        else:
            image_to_show = image
        cv2.imshow(self.window_name, image_to_show)
        if self.run_cv_waitkey and not do_not_run_waitkey:
            self._wait_for_key()
