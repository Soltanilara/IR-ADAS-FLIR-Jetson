import subprocess
import Jetson.GPIO as GPIO

GPIO_PORT=4

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(GPIO_PORT, GPIO.IN)

def check_power(power, pwr_cnt, fps):
    if GPIO.input(GPIO_PORT):
        pwr_cnt += 1
        if pwr_cnt == 1:
            print("Power failure detected. Please connect to AC adapter in 1 minute")
    else:
        if pwr_cnt != 0:
            print("Power restored")
            pwr_cnt = 0 # power restored
            
    if pwr_cnt > fps*60: # power not restored for 1 minute
        power = False

    return power, pwr_cnt 

def shutdown():
    print('Power failure. System shut down in 1 minute')
    # subprocess.run('sudo sync; shutdown -h +1', shell=True)