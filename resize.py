# import cv2
# import os, shutil, sys 
# from shutil import move
 

# file = '/home/lab/Documents/python/ClumsyChickens/datasets/stage1_train/'

# for dirPath, dirNames, fileNames in os.walk(file):
#     # print (dirPath)
#     for f in fileNames:
#         # print(os.path.join(dirPath, f))
#         # print(os.path.join(dirPath, f).split("/")[-1].split(".jpg")[0])
#         img = cv2.imread(os.path.join(dirPath, f))
#         pic = cv2.resize(img, (512, 288), interpolation=cv2.INTER_CUBIC)
#         cv2.imwrite(os.path.join(dirPath, f), pic)
#         # try:
#         #     os.makedirs(os.path.join(dirPath, os.path.join(dirPath, f).split("/")[-1].split(".jpg")[0]))
#         #     os.makedirs(os.path.join(os.path.join(dirPath, os.path.join(dirPath, f).split("/")[-1].split\
#         #         (".jpg")[0]), 'images'))
#         #     print(os.path.join(os.path.join(dirPath, os.path.join(dirPath, f).split("/")[-1].split(".jpg")\
#         #         [0]),'./images/'),f)
#         #     shutil.move(os.path.join(dirPath, f),os.path.join(os.path.join(dirPath, os.path.join(dirPath, f)\
#         #         .split("/")[-1].split(".jpg")[0]),'./images/'),f)            
#         # except FileExistsError:
#         #     print('error')

#         # img = cv2.imread(os.path.join(dirPath, f))
#         # pic = cv2.resize(img, (512, 384), interpolation=cv2.INTER_CUBIC)
#         # cv2.imwrite(os.path.join(dirPath, f), pic)



