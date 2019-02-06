# donkeyRL
Steps (assuming you have python 3.6 and required libraries):
1. Download unity
2. Get repository (donkey branch) here https://github.com/tawnkramer/sdsandbox/tree/donkey  (he also explains required libraries)
3. Store sim_rl_server_meetup.py in the src directory of above (i.e. *\sdsandbox\src)
4. Run unity simulator by loading the road_generator scene (i.e. road_generator.unity)
5. Click unity left road generator panel click->donkey->camerSensorBase->cameraSensor, then look for width/height and change to 320/240
6. Click play button and then choose the run NN steering with websockets
   
          ../source/model  (predict with model? t/f) false runs RL  (save jpg's of run? t/f) 
To RUN: python sim_rl_server_meetup.py ../outputs/mymodelR3 False True     (this would run the simulation solving actions with RL and store jpgs of the run)    

