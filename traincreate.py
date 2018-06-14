import numpy as np
import cv2
import  config
import os
listread=config.datalistpath
def framecreate():

  sourcepath="E:/data2/"
  outputpath="E:/testoutput/"
  for root, child, files in os.walk(sourcepath):

          for each in files:
           if not os.path.exists(outputpath+each[0:-4]):
                 os.mkdir(outputpath+each[0:-4])
           cap = cv2.VideoCapture(sourcepath+each)
           os.chdir(outputpath+each[0:-4])
           count = 1
           sucess=True
           while (sucess):
               sucess, frame = cap.read()
               if sucess==False:
                   print(each[0:-4]+"done")
               cv2.imwrite( str(count) + '.jpg', frame)  # 存储为图像
               count = count + 1

           cap.release()








if __name__ == '__main__':
    print('processing!')
    framecreate()
