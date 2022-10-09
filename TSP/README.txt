README

arguments:
-path       # Enter path to the TSP file (include tsp)
            # Default is "Instances/a280.tsp"

-alg        # Enter number for alg 
            #   -1: All (Default)
            #    0: First ascent from rand
            #    1: First ascent from randx4000
            #    2: Steepest ascent from rand
            #    3: Steepest ascent from randx4000 
            #    4: Random ascent from rand
            #    5: Random ascent from randx4000

-s          # Include to plot paths found 

-l          # Include to plot path lengths 

Run code (all algorithms):
python3 run.py 

Run code (Firs ascent from rand on berlin52 problem):
python3 run.py -path Instances/berlin52.tsp -alg 0 

Run code (Firs ascent from rand on berlin52 problem and show path length):
python3 run.py -path Instances/berlin52.tsp -alg 0 -l

(The above assumes all problem Instances are found in an Instances folder)