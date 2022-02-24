import pathlib
import copy
import scipy.integrate as integrate
from scipy.stats import norm
import matplotlib.pyplot as plt
from argparse import Namespace


class GaussianDistribution: 
    _max_pdfs = []
    
    def __init__(self, fname):
        self._crash_percents, self.incremental_scores = self.read_data(fname)
        self._tool = fname

    
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

    def _incremental_scores(self): 
        'Plots the scores measured each 10k frames by the fuzzer'
        
        if len(self.incremental_scores) == 1 or  len(self.incremental_scores) == 0 :
            print("No incremental score received.")
            return 0

        time = list(range(0, (len(self.incremental_scores))*10000, 10000))
        
        plt.plot(time, self.incremental_scores)
        plt.xlabel('Time frames (10k unit)')
        plt.ylabel('Coverage score score') 
        plt.title(f"Coverage Score improvement of {self._tool}")
        plt.savefig(f"plots/{self._tool}_coverage_improve.png")
        

    def _max_pdf(self):
        ''
        if len(self._max_pdfs) == 0:
            for i in range(0, 101):
                self._max_pdfs.append(self.pdfs(i))

        plt.plot(range(0, 101), self._max_pdfs)
        plt.xlabel('Projection of State Space')
        plt.ylabel("Max Score")
        plt.title('Maximum Coverage Score ')
        plt.savefig(f"plots/{self._tool}_max_coverage_score.png")



def main(): 
    """
    args = argparse.ArgumentParser(description='Output Coverage Score Plots for Fuzz Testing Frameworks')
    parser.add_argument("--save-plots", action='store_true', default=False, help="Save plot files to disk.")
    parser.add_argument("--random", action='store_true')
    """


    """
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
    """ 
   
    pathlib.Path('./plots').mkdir(parents=True, exist_ok=True)

    for tool in [ "cpsfuzz_data", "random_data", "hypothesis_data", "atheris_data"]:
        gd = GaussianDistribution(tool)

        print(f"Total crashes: {gd._get_crash_count()} \n")
        print(f"Score {tool}: {gd.pdf_integral()[0]} \n")

        gd._incremental_scores()
        gd._max_pdf()


if __name__ == "__main__":
    main()




