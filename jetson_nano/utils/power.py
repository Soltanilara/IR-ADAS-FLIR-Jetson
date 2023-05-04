import subprocess
import Jetson.GPIO as GPIO
import PySimpleGUI as sg

GPIO_PORT=4

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(GPIO_PORT, GPIO.IN)

def check_power(power, pwr_cnt, fps):
    if GPIO.input(GPIO_PORT):
        pwr_cnt += 1
        if pwr_cnt == 1:
            sg.popup_timed('Power failure detected. Please connect to AC adapter', button_type=5, auto_close=True, auto_close_duration=3, non_blocking=True, no_titlebar=True, background_color='red', text_color='white')
    else:
        if pwr_cnt != 0:
            pwr_cnt = 0 # power restored
            sg.popup_timed('Power Restored', button_type=5, auto_close=True, auto_close_duration=3, non_blocking=True, no_titlebar=True, background_color='green', text_color='white')
    
    if pwr_cnt > fps*15: # power not restored for 30 seconds
        power = False

    return power, pwr_cnt 

def shutdown():
    sg.popup_timed('Power failure. System shut down in 1 minute', button_type=5, auto_close=True, auto_close_duration=60, non_blocking=True, no_titlebar=True, background_color='red', text_color='white')
    subprocess.run('sudo sync; shutdown -h +1', shell=True)
