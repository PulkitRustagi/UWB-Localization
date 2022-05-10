#! /usr/bin/env python
# coding=utf-8

import matplotlib

matplotlib.use('Qt5Agg')

import rospy
from gtec_msgs.msg import Ranging
from live_plotter import LivePlotter
import sys


class RangePlotter(object):
    """
    A class to help plot the UWB range data
    """

    def __init__(self, tags=None, n=-1):
        """
        Setups the RangePLotter class
        @param tags: The tags to plot in the range plotter. If it is None, plot everything
        @param n: The n latest range measurements
        """
        toa_ranging = '/gtec/toa/ranging'

        # self.live_plotter = LivePlotter(window_name="Range Drawer", lengend_loc="upper left", font_size='xx-small')
        self.live_plotter = LivePlotter(window_name="Range Drawer")

        self.n = n

        if tags is None:
            self.tags = None
        else:
            self.tags = set(map(int, tags))

        ranging_sub = rospy.Subscriber(toa_ranging, Ranging, callback=self.add_ranging)

    def add_ranging(self, msg):
        # type: (Ranging) -> None
        """
        Adds the range data if it part of the accepted tag ids to the live plotter
        @param msg: the Ranging msg
        """

        anchor_id = msg.anchorId
        tag_id = msg.tagId

        if self.tags is None or tag_id in self.tags:
            label = "tag{}_anchor{}".format(tag_id, anchor_id)

            time = rospy.Time.now().to_sec()

            self.live_plotter.add_data_point(label, time, msg.range)

    def run(self):
        """
        Running the live plotter
        """
        self.live_plotter.show()


if __name__ == "__main__":
    rospy.init_node("data_drawer_node")

    myargv = rospy.myargv(argv=sys.argv)[1:]

    print(myargv)

    if len(myargv) == 0:
        myargv = None

    data_plotter = RangePlotter(myargv)
    data_plotter.run()
