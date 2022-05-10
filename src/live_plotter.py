# coding=utf-8
from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class LivePlotter(object):
    """
    A helper class for creating live plotting matplotlib windows
    """

    def __init__(self, update_interval=1000, alpha=1.0, window_name=None, time_removal=None, lengend_loc='best',
                 font_size=None):
        """
        Setups the live plotting window
        @param update_interval: How fast the animated window updates its values
        @param alpha: The alpha value of lines added if there are many to make things a little easier to view
        @param window_name: The matplotlib window name
        @param time_removal:  How long before you want to remove the legend, None to keep it indefinitely
        @param lengend_loc: The location of the legend as defined by matplotlib parameter choices
        @param font_size: The font size of the legend as defined by matplotlib parameter choices
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.tight_layout()
        if window_name is not None:
            self.fig.canvas.set_window_title(window_name)

        self.update_interval = update_interval

        self.alpha = alpha
        self.data = dict()
        self.objects = dict()
        self.ani = None
        self.time_removal = time_removal
        self.legend_loc = lengend_loc
        self.font_size = font_size

    def add_data_point(self, object_name, x, y):
        """
        Adds data to the live plotting window
        @param object_name: The line object to add the data to. If it does not exist a new object will be created
        otherwise the data will just be appended to
        @param x: the x data to add
        @param y: the y data to add
        """
        if object_name not in self.data:
            self.data[object_name] = {
                "x": [],
                "y": []
            }

            line, = self.ax.plot([], [], label=object_name, alpha=self.alpha)

            self.objects[object_name] = line

        self.data[object_name]["x"].append(x)
        self.data[object_name]["y"].append(y)

        self.objects[object_name].set_xdata(self.data[object_name]['x'])
        self.objects[object_name].set_ydata(self.data[object_name]['y'])

    def func_animate(self, i):
        """
        The animation function called at each interval step
        @param i: the current index of the animation
        @return: the objects to draw
        """
        try:
            if self.time_removal is None or i < self.time_removal:
                # if self.time_removal is not None:
                #     print(i)

                self.ax.legend(loc=self.legend_loc, fontsize=self.font_size)
            else:
                self.ax.legend_ = None
            self.ax.relim()
            self.ax.autoscale_view()
        except ValueError:
            print("Error graphing")

        return self.objects.values()

    def show(self):
        """
        Shows and starts the live plotter func loop
        """
        self.ani = FuncAnimation(self.fig, self.func_animate, interval=self.update_interval)

        plt.show()


if __name__ == '__main__':
    plotter = LivePlotter()
    plotter.show()
