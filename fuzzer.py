import os
import sys
import random
import math
import copy
import numpy as np

import yaml
import scipy.integrate as integrate
from scipy.stats import norm
from argparse import Namespace

import smooth_blocking_vs_blocking  as sbb



class CrashDistribution:


    def __init__(self):

        self._crash_percents = []

        # config
        with open("config.yaml") as f:
            conf_dict = yaml.load(f, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)
        lanes = np.loadtxt(conf.lanes_path, delimiter=conf.lanes_delim, skiprows=conf.wpt_rowskip)
        center_lane_index = 1
        self.center_lane = lanes[:, center_lane_index*3:center_lane_index*3+2]


    def _update(self, x, y):

        percent = self. _percent_completed(x, y)
        self._crash_percents.append(percent)
      
        return percent


    def _pdfs(self, x):

        tmp = []
        
        if len(self._crash_percents) == 0: 
            return 0
   
        'Find the closest node to x, pdf() will be maximum at that point'

        closest =  self._crash_percents[min(range(len(self._crash_percents)), key = lambda i: abs(self._crash_percents[i]-x))]

        return norm.pdf(x, closest, 1)



    def _pdf_integral(self):
        """"""
        res =  integrate.quad(self._pdfs, 0, 100, limit=100)
        return res[0]


    def _get_crash_percents(self):

        return self._crash_percents


    def _percent_completed(self, own_x, own_y):
        'find the percent completed in the race just using the closest waypoint'

        num_waypoints = self.center_lane.shape[0]

        min_dist_sq = np.inf
        rv = 0

        # min_dist_sq is the squared sum of the two legs of a triangle
        # considering two waypoints at a time

        for i in range(len(self.center_lane) - 2):
            x1, y1 = self.center_lane[i]
            x2, y2 = self.center_lane[i + 1]

            dx1 = (x1 - own_x)
            dy1 = (y1 - own_y)

            dx2 = (x2 - own_x)
            dy2 = (y2 - own_y)


            dist_sq1 = dx1*dx1 + dy1*dy1
            dist_sq2 = dx2*dx2 + dy2*dy2
            dist_sq = dist_sq1 + dist_sq2

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                # 100 * min_index / num_waypoints

                rv = 100 * i / num_waypoints

                # add the fraction completed betwen the waypints
                dist1 = math.sqrt(dist_sq1)
                dist2 = math.sqrt(dist_sq2)
                frac = dist1 / (dist1 + dist2)

                assert 0.0 <= frac <= 1.0
                rv += frac / num_waypoints

        return rv




class Fuzzer:

    def __init__(self, fn):
        self.runs = 0
        self.total_frames = 0
        self.last_seen_frames = 0
        self.incremental_scores = [0]
        self.cd = CrashDistribution()
        
        open(fn, 'w').close() # cleanup the crash input file 
        self.crash_file = open(fn, 'w')


    def run(self, inp):

        percent = 0
        done, time_frames, x, y  = sbb.run(inp)

        self.total_frames += time_frames
        self.runs  = self.runs + 1

        print("run: ", self.runs, " frames: ", self.total_frames)

        if done == 1:
            percent = self.cd._update(x, y)
            wc = self.crash_file.write(str(inp) + "\n")
            self.crash_file.flush()



        if (self.total_frames - self.last_seen_frames) > 10000:
            score = self.cd._pdf_integral()
            self.incremental_scores.append(score)
            self.last_seen_frames = self.total_frames
       

        return self.runs, self.total_frames, done


    def get_crash_percents(self):

        return self.cd._get_crash_percents()


    def get_total_scores(self):

        return self.incremental_scores


    def save(self, fn):
       
        percents = copy.copy(self.get_crash_percents())
        scores = copy.copy(self.get_total_scores())
 
        open(fn, 'w').close() # clean up the file
    
        f = open(fn, "a")
        f.write(str(percents))
        f.write("\n")
        f.write(str(scores))
        f.flush()
        f.close()

    

