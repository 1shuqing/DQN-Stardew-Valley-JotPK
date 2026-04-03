import time

import cv2
import pyautogui
import numpy as np
import torch
from ultralytics import YOLO

def catch_screen(x,y,width,height):
    #截图
    sc = pyautogui.screenshot(region=(x,y,width,height))
    f = np.array(sc)
    f = cv2.cvtColor(f,cv2.COLOR_RGB2BGR)
    return  f




#
# yolo_model = YOLO(r"runs/detect/train/weights/best.pt")
# def detect_yolo(screenshot):
#     results = yolo_model(screenshot,verbose=False)
#     player_box = None
#     enemy_boxes = []
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             cls = int(box.cls)
#             if cls == 0 and player_box is None:
#                 player_box = box.xyxy.cpu().numpy()[0].tolist()
#             elif cls == 1 and len(enemy_boxes)<20:
#                 enemy_boxes.append(box.xyxy.cpu().numpy()[0].tolist())
#
#     max_size = max(650,650)
#     player_feat = [0.0]*4 if player_box is None else [x/max_size for x in player_box]
#     enemy_feat = []
#     for i in range(20):
#         if i<len(enemy_boxes):
#             enemy_feat.extend([x/max_size for x in enemy_boxes[i]])
#         else :
#             enemy_feat.extend([0.0]*4)
#     #归一
#     enemy_count = len(enemy_boxes)/20
#
#     #拼接特征
#     yolo_feat = player_feat + enemy_feat +[enemy_count]
#     yolo_state = torch.tensor(yolo_feat,dtype=torch.float).to("cuda")
#     yolo_state = yolo_state.unsqueeze(0)    #w
#     return yolo_state
#
# # tensor([[0.2766, 0.2208, 0.3350, 0.2788, 0.3903, 0.4864, 0.4522, 0.5482, 0.2808, 0.3173, 0.3428, 0.3789, 0.5191, 0.4680, 0.5811, 0.5279, 0.4088, 0.4266, 0.4695, 0.4861, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
# #          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
# #          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2000]], device='cuda:0')
#
# #死亡场景
# # tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
# #        device='cuda:0')
#
#
#
# def extract_boxes_with_classes(results):
#     """
#     从YOLO预测结果中提取框坐标和类别ID，整理为指定格式
#     :param results: YOLO.predict()的返回值
#     :return: [(类别ID, (x左上角, y右下角)), ...]
#     """
#     target_format = []
#     # 遍历每张图片的结果（适配单/多图场景）
#     for result in results:
#         if result.boxes is None:  # 无检测结果时直接返回空列表
#             continue
#         # 提取所有框的坐标（xyxy格式：x1, y1, x2, y2）
#         boxes_xyxy = result.boxes.xyxy.cpu().numpy()
#         # 提取所有框的类别ID（转为整数）
#         class_ids = result.boxes.cls.cpu().numpy().astype(int)
#
#         # 遍历每个检测框，整理格式
#         for cls_id, box in zip(class_ids, boxes_xyxy):
#             x1, y1, x2, y2 = box  # 拆分坐标：x1=左上x, y1=左上y, x2=右下x, y2=右下y
#             # 按你的要求：(类别ID, (x左上角, y右下角))
#             target_format.append((int(cls_id), (float(x1), float(y2))))
#     return target_format
#
# def yolo_detect(img,model):
#     #将截取的图片用训练好的yolo检测
#     result = model.predict(cs, save=True, conf=0.3)
#     return result

#游戏窗口在屏幕左上角时的时间读条位置
x, y, width, height = 376,88,560,10

# cs = catch_screen(x, y, width, height)
# cv2.imshow("time",cs)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__ == '__main__':
    model = YOLO(r"runs/detect/train2/weights/best.pt")

    cs = catch_screen(300, 100, 650, 600)
    # print(detect_yolo(cs))
    yolo_detect(cs,model)

    # print(extract_boxes_with_classes(result))
