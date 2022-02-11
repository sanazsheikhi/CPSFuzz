
"""
CPSFuzzer is a project for fuzz testing cyber physical systems.This work is the
implemention of CPSFuzzer for autonomous racing project F110th. It is consist 
of the following classes:

1- Fuzzer: main class interacting with the fuzz testing classes, F110 program, 
           and the cps score class

2- Scheduler: is part of the fuzz testing mechanism. It chooses a seed to be 
              mutated for new test case generation. It considers different 
              partitions of the state space and the cps score assigned to them 
              to peak a relevant seed accordingly. It also, maintains local see
              corpuses for each section respectively and takes care of fuzzing 
              time devision between them to avoid turning them into local minimum 
              thats eating up the major amount of fuzzing cycles whitout contributing
              to cps score. 

3- Mutator: is responsible for mutating a seed and generating a new test case. 
            There variety of mutation operation such as randomly change a range 
            of data in the seed, flipping all values in a range, etc. The key is
            that these operation are not bit/byte level but item level.


4- CrashDistribution: keeps track of the events happening in the objective space 
                      and calculates cps score based on the cumulative probability 
                      distribution.
"""

import os
import sys
import random
import numpy as np
import smooth_blocking_vs_blocking  as sbb
import copy
import math
import yaml
import scipy.integrate as integrate
from scipy.stats import norm
from argparse import Namespace
import time


num_interval = 4 # global var



class Seed:
    
    def __init__(self, data):    
        self.data = data 
        self.child = 1 # energy = 1 / child
        self.freq = 0

    def get_data(self):
        return self.data


class Mutator:
     
    def __init__(self):

        # Use len of data not interval len
        global num_interval
        self.interval_len = 80 // num_interval


    def mutate_corpus(self, data, interval):
 
        'Randomly mutates  parts of the seed related to the chosen interval'

        if len(data) == 0:
            return data

        s = interval * self.interval_len
        e = s +  self.interval_len - 1 

        
        if s >= len(data):
            return data
        if e >= len(data):
            e = len(data) - 1 
         
        """Only one mutation is not helpful in practice and does not make 
         serious change in input.Like other fuzzers such as AFL, we should 
         perform several mutations on an input to make a difference"""
    
        sub = [random.choice([-1,0,1]) for _ in range(e-s+1)]
        data[s:e+1]= sub

        return data


    def mutate_1st_half(self, data, interval):
    
        'Randomly mutates first half of the part related to the interval in seed'

        if len(data) == 0:
            return data

        s = interval * self.interval_len
        e = s +  self.interval_len//2 


        if s >= len(data): 
            return data
        if e >= len(data):
            e = len(data) - 1
   
 
        sub = [random.choice([-1,0,1]) for _ in range(e-s+1)]
        data[s:e+1]= sub

        return data



    def mutate_2nd_half(self, data, interval):
    
        'Randomly mutates second half of the part related to the interval in seed' 

        if len(data) == 0:
            return data

        s = interval * self.interval_len + self.interval_len//2
        e = s +  self.interval_len//2 + 1


        if s >= len(data): 
            return data
        if e >= len(data):
            e = len(data) - 1

    
        sub = [random.choice([-1,0,1]) for _ in range(e-s+1)]
        data[s:e+1]= sub

        return data

   

    def mutate_pre_interval(self, data, interval):
    
        'Start mutating from second half of previous interval'   
     
        if len(data) == 0:
            return data
 
        if interval == 0:
            interval += 1

        s = (interval-1) * self.interval_len + self.interval_len//2
        e = s +  self.interval_len//2 + self.interval_len


        if s >= len(data): 
            return data
        if e >= len(data):
            e = len(data) - 1

    
        sub = [random.choice([-1,0,1]) for _ in range(e-s+1)]
        data[s:e+1]= sub

        return data



    def mutate_reverse(self, data, interval):
    
        'Resverse all the elements in the seed'

        if len(data) == 0:
            return data

        data.reverse()
        return data



    def mutate_flip(self, data, interval):
        
        ' flip the elements in the seed'

        if len(data) == 0:
            return data

        s = interval * self.interval_len
        e = s +  self.interval_len - 1 

        
        if s >= len(data):
            return data
        if e >= len(data):
            e = len(data) - 1 
         

        for i in range(s,e+1):
            data[i] *= -1

        return data





    def mutate(self, data, interval):

        variants = []
        
        v = random.randint(0, 5)
        
        if  v == 0:
            variants.append(self.mutate_corpus(data, interval))
        elif v == 1:
            variants.append(self.mutate_1st_half(data, interval))
        elif v == 2:
            variants.append(self.mutate_2nd_half(data, interval))
        elif v == 3:
            variants.append(self.mutate_pre_interval(data, interval))
        elif v == 4:
            variants.append(self.mutate_reverse(data, interval))
        else:
            variants.append(self.mutate_flip(data, interval))

        return variants
        


    operators = ["insert_rand", "insert_multi_rand", "delete_rand", 
                 "delete_multi_rand", "update_rand", "update_multi_rand"]



    def insert_rand(self, data, s, e):

        'insert a random number i to a random position in interval [s,e]'

        i = random.randint(-1, 1)
        j = random.randint(s, e)
        data.insert(j, i)
        return data
    
    def insert_multi_rand(self, data, s, e):

        'insert a random number i to interval [s,e]'

        count = random.randint(2, self.interval_len/2)
        while(count > 0):
            i = random.randint(-1, 1)
            j = random.randint(s, e)
            data.insert(j, i)
            count -= 1
        return data
 
    def update_rand(self, data, s, e):

        'pdate the interval [s, e] with random number'

        i = random.randint(-1, 1)
        j = random.randint(s, e)
        if j < len(data):
            data[j] = i
        
        return data
    
    def update_multi_rand(self, data, s, e):

        'update the interval [s, e] randon number of times'

        count = random.randint(2, self.interval_len/2)
        while(count > 0):
            i = random.randint(-1, 1)
            j = random.randint(s, e)
            if j < len(data):
                data[j] = i
            count -= 1 
        
        return data
 
    def delete_rand(self, data, s, e):
        
        'delete from a random index'

        i = random.randint(s, e) 
        if i < len(data):
            del data[i]

        return data
 
    def delete_multi_rand(self, data, s, e):
   
       'delete the crash index'

       count = random.randint(2, self.interval_len/2)
       while(count > 0):
           j = random.randint(s, e)
           if len(data) > j:
               del data[j] 
           count -= 1 

       return data
        



class PowerSchedule():
    
 
    
    def __init__(self):
 
        self._cd = CrashDistribution();

        self.main_corpus = []
        global num_interval
        self.local_corpus = [[] for i in range(num_interval)]

      
        'Initialize the main corpus with non-crashing seeds from file'

        f = open("seeds", 'r')
        seeds = f.readlines() 
        self.init_main_corpus_size = len(seeds)
        for s in seeds:
            s = s.replace("[", "")
            s = s.replace("]", "")
            ls = list(map( int, s.split(',')))
            self.add_2_main_corpus(ls)
  

   
    def choose(self):
      
        """ This function chooses the interval with lowest cps score while takes 
          care of fairness that while one interval get most of the fuzzing cycles
          the others wont starve """
        
        global num_interval

        """Each seed can be chosen up to the threshold number of times.
         To avoid randomly giving the chance to one seed"""

        seed_freq_threshold = 2

        'Choosing the interval with lowest score' 

        interval = self._cd._lowest_score_interval() 
           
      
        'Find one seed either from local or global corpus based frequency'

        while (True):

            if len(self.local_corpus[interval]) != 0:
                tmp = self.local_corpus[interval]
                corpus_id = interval
            else:
                tmp = self.main_corpus
                corpus_id = -1
         

            seed_id = self.get_youngest_seed(tmp)
            seed = tmp[seed_id]


            'Removing aged seeds. As they are not generating new seeds anymore'

            if corpus_id != -1 and seed.freq >= seed_freq_threshold:
                self._remove_from_local_corpus(corpus_id, seed_id)
                continue

            seed.freq += 1
            tmp[seed_id] = seed
         
            break

        return seed.data, corpus_id, interval, seed_id


        
    def add_2_main_corpus(self, inp):
        
        'adding a non-crash seed to interval corpus'    
        
        seed = Seed(inp) 
        self.main_corpus.append(copy.deepcopy(seed))    
       

    def add_2_local_corpus(self, crash_percent, inp):

        'add new seed to corpus'

        seed = Seed(inp)
        global num_interval
        corpus_id = int(crash_percent // (100//num_interval))

        self.local_corpus[corpus_id].append(copy.deepcopy(seed))

        'remove old seed from corpus'

        """tmp = self.local_corpus[corpus_id]
        #seed_id = tmp.index(min(range(len(tmp)), key = lambda i: tmp[i].freq))
        seed_id = self.get_youngest_seed(tmp)
        self._remove_from_local_corpus(corpus_id, seed_id)""" # deletes the only seed for us!

 

    def _remove_from_main_corpus(self, seed_id):        

       'Remove a seed from main corpus'       

       if len(self.main_corpus) > self.init_main_corpus_size:
           del self.main_corpus[seed_id]
       else:
          print("Can not remove seeds from main corpus")
   


    def _remove_from_local_corpus(self, corpus_id, seed_id):        
       
       'Remove a seed from a local corpus'      

       del self.local_corpus[corpus_id][seed_id]
 

    def get_youngest_seed(self, corpus):
    
        'finds the leasr frequently used seed from the corpus'

        if len(corpus) == 0:
            return -1

        index = 0
        val = corpus[0].freq
        for i in range(len(corpus)):
      
            if corpus[i].freq < val:
                val = corpus[i].freq
                index = i

        return index
 

    def get_total_score(self):

        return self._cd.get_total_score()


    def get_crash_percents(self):

        return self._cd.get_crash_percents()



class CrashDistribution:

    global num_interval

    _crash_percents = [[]] * num_interval
    interval_scores = [0] * num_interval
    _current_interval = []
    accounting_intervals = [0] * num_interval
   
    def __init__(self):

        'Read F110th env config'
        # config
        with open("config.yaml") as f:
            conf_dict = yaml.load(f, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)
        lanes = np.loadtxt(conf.lanes_path, delimiter=conf.lanes_delim, skiprows=conf.wpt_rowskip)
        center_lane_index = 1
        self.center_lane = lanes[:, center_lane_index*3:center_lane_index*3+2]



    def _update_interval_crash_percent(self, x, y): 
       
        'Updates the information of each interval upon new crash arrival' 
       
        percent = self._percent_completed(x, y)

        global num_interval
        interval_len = int(100 // num_interval)
        interval = int(percent //interval_len)
        tmp = copy.deepcopy(self._crash_percents[interval])
        tmp.append(percent)
        self._crash_percents[interval] = copy.deepcopy(tmp)
        
        return percent, interval



    def _pdfs(self, x):
        
        """
            Calculate probability distribution function value for the objective 
            state space. In the F110 project, we have projected the objective spac 
            to one dimention, percent of the race lap completed. We consider it to 
            be 0 to 100 percent. Each crash position becomes the mean of a pdf kernel 
            function and the commulative sum of the kernel functions shows the coverage 
            of objective space 
        """

        tmp = []
      
        'Find the closest node to x, pdf() will be maximum at that point'
       
        if len(self._current_interval) == 0:
            return 0
        elif len(self._current_interval) == 1:
            closest = self._current_interval[0] 
        else:
            closest =  self._current_interval[min(range(len(self._current_interval)), key = lambda i: abs(self._current_interval[i]-x))]

        return norm.pdf(x, closest, 1)



    def _pdf_integral(self):
        
        """ Calculates commulative sum of kernel pdf functions to measure coverage 
         of the objective state space """

        return integrate.quad(self._pdfs, 0, 100, limit=100)



    def _lowest_score_interval(self):

        """ Finding the inverval with lowest pdf score. To avoid one interval 
          turning into local minima we keep track of how many times each 
          interval has been chosen in one path """


        threshold = 2

        tmp = copy.deepcopy(self.interval_scores)       
        tmp.sort() # ascendiing order index 0 == minimum value
        out = -1
       
        for i in range(len(tmp)):
            index = self.interval_scores.index(tmp[i])
            if (self.accounting_intervals[index] > threshold):
                continue
            else:
               out = index
               self.accounting_intervals[index] += 1
               break
               
       
        'all intervals have been explored threshold times. So, reset their profile'

        if out == -1:
            self.accounting_intervals[0:] = [0] * len(self.accounting_intervals)
            out = 0
            self.accounting_intervals[0] += 1


        return out
            


    def _update_interval_score(self, interval):

        'Update the score of intervals based on new crashes'
 
        self._current_interval = copy.deepcopy(self._crash_percents[interval])
        self.interval_scores[interval] = copy.deepcopy(self._pdf_integral()[0])



    def get_total_score(self):
 
        'Get the total score through the whole space'
       
        self._current_interval = copy.copy(self.get_crash_percents())
        
        return self._pdf_integral()[0]


    def get_crash_percents(self):

        percents = []
        
        for l in self._crash_percents:
            percents += l

        return percents



    def _percent_completed(self, own_x, own_y):

        'Find the percent completed in the race just using closest waypoint'

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
    
                rv = 100 * i / num_waypoints

                # add the fraction completed betwen the waypints
                dist1 = math.sqrt(dist_sq1)
                dist2 = math.sqrt(dist_sq2)
                frac = dist1 / (dist1 + dist2)

                assert 0.0 <= frac <= 1.0 
                rv += frac / num_waypoints

        return rv




class Fuzzer:

    last_seen_frames = 0
    new_crash_step_count = 0
    crash_count = 0
    total_frames = 0

    def __init__(self):
        self.ps = PowerSchedule()
        self.mu = Mutator()

        open("cpsfuzz_crash_cmd", 'w').close() # cleanup the crash input file
        self.crash_file = open("cpsfuzz_crash_cmd", 'w') # data for DBScan

        self.incremental_scores = [0]



    def _fuzz(self):
        
        """
        data : selected seed for mutation
        corpus: main or local corpus where seed is selected from
        interval: the least score interval that we are looking to mutate a seed for
        seed: id of the data seed being chosen for mutation
        """

        data, corpus, interval, seed = self.ps.choose()
        inp_set = self.mu.mutate(data, interval) 

        return [data, inp_set, corpus, interval, seed]
        
  
    def run(self):

        data, inp_set, corpus, interval, seed = self._fuzz()

        if (inp_set is None or len(inp_set) == 0):
            print("No Fuzzing Input")
            return
 
        for inp in inp_set:

            'Running the system under test, F110th'

            is_crash, time, x_e, y_e  = sbb.run(inp)
       
            'Number of cycles used during one run'

            self.total_frames += time
            percent = 0          

            'The mutated data led the program crash, accident in F110th scenario'

            if (is_crash == 1):
                percent, interval = self.ps._cd._update_interval_crash_percent(x_e, y_e)
                self.ps._cd._update_interval_score(interval)
            

                """If the seed was chosen from the main corpus (data) and mutation 
                   of that seed led to crash then move that seed (data) to a 
                   local corpus where potentially can cause  more crashed for 
                   respective interval"""

                if corpus == -1:
                    self.ps.add_2_local_corpus(percent, data)
                    self.ps._remove_from_main_corpus(seed)

            
                'store the crash input in file to be used by DBScan algorithm'

                wc = self.crash_file.write(str(inp) + "\n")
                self.crash_file.flush()
            
                self.crash_count += 1


            'If the mutated seed didnt make crash, add it to the main corpus'

            if is_crash == -1:
                self.ps.add_2_main_corpus(inp)

            

    def runs(self, count):

        r = 0
        while self.total_frames < count:

            print("run ", r, " frames: ", self.total_frames)
            r += 1
            self.run()
            
            'For calculating score improvement'

            if (self.total_frames - self.last_seen_frames) > 10000 :
                self.incremental_scores.append(self.ps.get_total_score())
                self.last_seen_frames = self.total_frames
                
        print("FINISHED")
                 
   

    def get_crash_percents(self):
    
         return self.ps.get_crash_percents()
   

    def get_total_scores(self):
        
        return self.incremental_scores


    def save(self, fn):
  
        """Save the crash data and incremental scores in a 
           file to be used for furthur analysis"""
 
        percents = copy.copy(self.get_crash_percents())
        scores = copy.copy(self.get_total_scores())
 
        open(fn, 'w').close() # clean up the file
    
        f = open(fn, "a")
        f.write(str(percents))
        f.write("\n")
        f.write(str(scores))
        f.close()
    
   
 
def main():

    print("Please enter the number of time_frames: ")
    iterations = int(input())
    fuzzer = Fuzzer()
    fuzzer.runs(iterations)
    fuzzer.save("cpsfuzz_data")




if __name__ == "__main__":

    main()
