import os
import cv2
path="E:/test/"
path2="E:/data2/"
#divide video into clips
for root, child, files in os.walk(path):
    #all files with be divided into 2s
    for each in files:
        if not os.path.exists(path2 + each[0:-4]):
            os.mkdir(path2 + each[0:-4])
        cap = cv2.VideoCapture(path + each)
        os.chdir(path2 + each[0:-4])
        sum=cap.get(7)/(cap.get(5)*3)
        fps = cap.get(5)
        size = ( int(cap.get(3)), int(cap.get(4)))
        count = 1
        total=int(cap.get(5)*3)
        while(count<=sum):
            videoWriter = cv2.VideoWriter(each[0:-4] + "-"+"clip"+str(count)+".avi",  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
            for num in range(0,total):
                success,frame = cap.read()
                if(frame is None):
                    break
                videoWriter.write(frame)

            videoWriter.release()
            count=count+1
        cap.release()
        print('done', each)