import time
import threading

import open3d as o3d


class Threading:
    def __init__(self):
        self.visualizer = None
        self.thread = threading.Thread(target=Threading.start_window, args=(self, ))

    def start_window(self):
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(window_name="whatever", width=600, height=400)
        render_options = self.visualizer.get_render_option()
        render_options.background_color = (0.498, 0.788, 1)
        render_options.light_on = True
        self.visualizer.run()

    def start_thread(self):
        self.thread.start()

    def cleanup(self):
        self.visualizer.destroy_window()
        self.thread.join()


def another_thread(argument):
    for i in range(5):
        print(argument, i)
        time.sleep(1)


if __name__ == "__main__":
    o = Threading()
    o.start_thread()
    time.sleep(5)
    o.cleanup()


    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window(window_name="whatever", width=600, height=400)
    #
    # render_options = visualizer.get_render_option()
    # render_options.background_color = (0.498, 0.788, 1)
    # render_options.light_on = True
    #
    # th = threading.Thread(target=another_thread, args=('another thread', ))
    # th.start()
    # time.sleep(2)
    #
    # visualizer.run()
    # visualizer.destroy_window()
    # th.join()
