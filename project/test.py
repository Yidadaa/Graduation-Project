from unity_arm_env import UnityArmEnv
import time

if __name__ == '__main__':
    test = UnityArmEnv()
    start = time.clock()
    i = 0
    while i < 1000:
        test.step([1,2,3])
        i += 1
        if i % 100 == 0:
            print(i)
    
    print(time.clock() - start)