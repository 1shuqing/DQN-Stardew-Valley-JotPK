import time
from pynput import keyboard
# from pynput.keyboard import Controller
c = keyboard.Controller()

def up():
    c.release('s')
    c.press('w')
    # c.release('w')
def down():
    c.release('w')
    c.press('s')
    # c.release('s')
def left():
    c.release('d')
    c.press('a')
    # c.release('a')
def right():
    c.release('a')
    c.press('d')
    # c.release('d')
def shootup():
    c.release(keyboard.Key.down)
    c.press(keyboard.Key.up)
    # c.release(keyboard.Key.up)
def shootdown():
    c.release(keyboard.Key.up)
    c.press(keyboard.Key.down)
    # c.release(keyboard.Key.down)
def shootleft():
    c.release(keyboard.Key.right)
    c.press(keyboard.Key.left)
    # c.release(keyboard.Key.left)
def shootright():
    c.release(keyboard.Key.left)
    c.press(keyboard.Key.right)
    # c.release(keyboard.Key.right)
def tools():
    c.press(keyboard.Key.space)
    c.release(keyboard.Key.space)

def release_all_key():
    for k in ['w','s','a','d',keyboard.Key.up,keyboard.Key.down,keyboard.Key.left,keyboard.Key.right]:
        try:
            c.release(k)
        except:
            print("释放按键失败!!!")
