import sys
import random
import fuzzer as fz


def main():
  
    lapLen = 80
    runs = 0  
    time_frames = 0

    print("Please enter number of time frames:")
    iterations = int(input())
    
    fuzzer = fz.Fuzzer("random_crash_cmd")
    inp = [random.choice([-1,0,1]) for i in range(lapLen+1)]
    
    #while runs < int(iterations):
    while time_frames < iterations:

        #count = random.randint(0, lapLen//4) # num of input cells to change 
        count = 1
        for i in range(0, count):
            index = random.randint(0, lapLen) # choose a random cell of the input for change
            inp[index] = random.choice([-1,0,1])

        runs, time_frames, _ = fuzzer.run(inp)

    fuzzer.save("random_data")
    return

    
if __name__ == "__main__":

    main()



