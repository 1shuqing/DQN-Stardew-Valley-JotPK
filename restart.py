import time
from pynput import keyboard

# 标记是否继续运行脚本（用于通过按键终止程序）
running = True


# 按键按下的回调函数
def on_press(key):
    global running
    try:
        # 按 q 键停止脚本
        if key.char == 'q':
            print("按下 Q 键，准备停止脚本...")
            running = False
            return False  # 停止监听
    except AttributeError:
        # 按enter 也停止脚本
        if key == keyboard.Key.enter:
            print("按下 enter 键，准备停止脚本...")
            running = False
            return False


# 你的原有重启函数（pynput 版本）
def restart():
    from pynput.keyboard import Controller
    keyboard_ctrl = Controller()

    time.sleep(3)
    print("似了，重开")
    keyboard_ctrl.press(keyboard.Key.esc)
    keyboard_ctrl.release(keyboard.Key.esc)
    time.sleep(0.5)
    # print("退出")

    keyboard_ctrl.press(keyboard.Key.space)
    keyboard_ctrl.release(keyboard.Key.space)
    time.sleep(0.5)
    # print("确认1")

    keyboard_ctrl.press(keyboard.Key.space)
    keyboard_ctrl.release(keyboard.Key.space)
    time.sleep(5)
    # print("确认2")



if __name__ == '__main__':
    # 启动后台按键监听线程（非阻塞）
    # listener = keyboard.Listener(on_press=on_press)
    # listener.start()
    #
    # print("5秒后开始执行重启脚本（按 Q 或 ESC 可终止）...")
    time.sleep(5)

    # 只有 running 为 True 时才执行脚本
    # while running:
    restart()
          # 执行完一次后等待2秒，可根据需求调整

    # 停止监听线程
    # listener.stop()
    # print("脚本已终止")