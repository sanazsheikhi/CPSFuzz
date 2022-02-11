import atheris
import sys
import fuzzer as fz
           

 
global fuzzer 
fuzzer = fz.Fuzzer("atheris_crash_cmd") 
global iterations

def TestOneInput(data):

    
    lapLen = 80
    fdp = atheris.FuzzedDataProvider(data)
    inp = fdp.ConsumeIntListInRange(lapLen,-1,1)

    global iterations
    global fuzzer
    runs, time_frames, _  = fuzzer.run(inp)

    #if runs == iterations:
    if time_frames > iterations:
        fuzzer.save("atheris_data")
        raise Exception("Stop Atheris fuzzer") # No ther way to stop atheris


def main():

    global iterations
    print("Please enter number of time frames:")
    iterations = int(input())

    try:

        atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
        atheris.Fuzz()

    except Exception as e:
        
        print(e)
         


if __name__ == "__main__":

    main()

