import os
import sys
import copy
import math
import numpy as np
import scipy.integrate as integrate
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib.pyplot as plt
from argparse import Namespace
import yaml




class GaussianDistribution:
    
    _max_pdfs = []
    
    def __init__(self, fname):

        self._crash_percents, self.incremental_scores = self.read_data(fname) 

    
    def read_data(self, fn):
    
        f = open(fn,'r')
        content = f.readlines()
        f.close() 

        'First line: percents, second line: scores'
        'covert string list to float list'

        content0 = content[0]
        content1 = content[1]
        content0 = content0.replace('[', "")
        content0 = content0.replace(']', "")
        content0 = content0.replace(' ', "")
        content0 = content0.replace('\n',"")
        l0 = content0.split(',')
        content1 = content1.replace('[', "")
        content1 = content1.replace(']', "")
        content1 = content1.replace(' ', "")
        l1 = content1.split(',')        
  
        percents = []
        scores = []

        for i in l0:
            percents.append(float(i))

        for i in l1:
            scores.append(float(i))

        return percents, scores
               

    def _get_crash_count(self):

        return len(self._crash_percents)

    def pdfs(self, x):
    
        if len(self._crash_percents) == 0:
            print("No crash data received")
            return 0

        closest =  self._crash_percents[min(range(len(self._crash_percents)), key = lambda i: abs(self._crash_percents[i]-x))]
 
        #return norm.pdf(x, closest, 1)
        return norm.pdf(x, closest, 1)
        
         

    def pdf_integral(self, l=0):
        """"""
        a = []
        b = []
        if l != 0:
            a = copy.deepcopy(self._crash_percents[0:l])
            b = copy.deepcopy(self._crash_percents)
            self._crash_percents.clear()
            self._crash_percents = copy.deepcopy(a)

        score = integrate.quad(self.pdfs, 0, 100, limit=100) 

        if l != 0: 
            self._crash_percents.clear()
            self._crash_percents = copy.deepcopy(b)
      
        return score
 


    def _incremental_scores(self, lb):
    
        'Plots the scores measured each 10k frames by the fuzzer'
        
        if len(self.incremental_scores) == 1 or  len(self.incremental_scores) == 0 :
            print("No incremental score received.")
            return 0


        time = list(range(0, (len(self.incremental_scores))*10000, 10000))
        
        plt.plot(time, self.incremental_scores)
        plt.xlabel('Time frames (10k unit)')
        plt.ylabel('Coverage score score') 
        plt.title('Coverage Score improvemnt of ' + lb)
        plt.show()
        


    def _max_pdf(self):
        ''
        if len(self._max_pdfs) == 0:
            for i in range(0, 101):
                self._max_pdfs.append(self.pdfs(i))

        #print("_max_pdfs: ", self._max_pdfs)

        plt.plot(range(0, 101), self._max_pdfs)
        plt.xlabel('projection of state space')
        plt.ylabel('max score')
        plt.title('Maximum coveragescore ')
        plt.show()




def main():

    print("Pleas enter the tool number:\n")
    print("1. CPSFuzz  2.Random fuzzing  3.Hypothesis  4.Atheris")
    num = int(input())
   
    if num == 1:
        tool = "cpsfuzz_data"
    elif num == 2:
        tool = "random_data"
    elif num == 3:
        tool = "hypothesis_data"
    elif num == 4:
        tool = "atheris_data"
    else:
        print("Wrong tool")
        return


   
    gd = GaussianDistribution(tool)
    print("\n\n")
    print("Total crashes: ", gd._get_crash_count())
    print("\n\n")
    print("score ", tool,":", gd.pdf_integral()[0])
    print("\n\n")
    gd._incremental_scores(tool)
    gd._max_pdf()

 
main()




