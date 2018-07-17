from unity_arm_env import UnityArmEnv
import time
from gd import Decide

if __name__ == '__main__':
    test = UnityArmEnv()
    start = time.clock()
    brain = Decide()
    i = 0
    while i < 1000:
        state = test.step([1,2,3])
        i += 1
        if i % 100 == 0:
            print(i)
            print(state)
    
    print(time.clock() - start)