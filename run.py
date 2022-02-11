import sys
import os
sys.path.append('/home/sanaz/Documents/Research/f1tenth_gym/ArtifactEvaluation')
import MyFuzzer as fu
#import crash_distribution as cd



def run():
    fuzzer = fu.Fuzzer()
    fuzzer.runs(10)
    cl = fuzzer.get_crash_list()
    print("crash list: ", cl)

run()
