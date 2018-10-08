__author__ = 'vmartinezf'


import threading, time
from datetime import datetime

time_cycle = 50 #ms

class ThreadNetwork(threading.Thread):
    def __init__(self, network):
        self.network = network

        threading.Thread.__init__(self)

    def run(self):

        while(True):
            start_time = datetime.now()
            self.network.predict()

            finish_Time = datetime.now()

            dt = finish_Time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0

            if(ms < time_cycle):
                time.sleep((time_cycle-ms) / 1000.0)