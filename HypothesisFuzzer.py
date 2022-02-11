import sys
import fuzzer as fz
from hypothesis import given, settings, strategies as st



global iterations
global fuzzer
global time_frames

strategies = [st.just(1), st.just(-1), st.just(0)]
@given(ls=st.lists(st.one_of(*strategies), min_size=80, max_size=80))
@settings(deadline=200000000, max_examples=500)

def hypo(ls):
   
    global iterations
    global fuzzer
    global time_frames


    if time_frames >= iterations:
        
        fuzzer.save("hypothesis_data")
        print("\nFinshed fuzz testing. Exiting ...\n")
        exit(0)


    runs, time_frames, done = fuzzer.run(ls)

    'property:no crash'

    assert done != 1, 'Crash detected' 





def main():
     
    global iterations
    global fuzzer
    global time_frames

    time_frames = 0

    print("Please enter the number of time frames:")
    iterations = int(input())

    fuzzer = fz.Fuzzer("hypothesis_crash_cmd")

    while(True):
        try:

            hypo()

        except AssertionError as e:

            print(e)
            continue



if __name__ == "__main__":

    main()
