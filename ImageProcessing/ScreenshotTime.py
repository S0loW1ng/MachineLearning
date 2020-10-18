import keyboard
import pyautogui
import sys
name = 0
while True:
    try:
        
        if keyboard.is_pressed('p'):
           print (pyautogui.position())
           im = pyautogui.screenshot("Flat"+str(name) + ".png",region=(661,461,1150,400))
           name = name +1


        else:
            pass
    except Exception as e:
        print (sys.exc_value)