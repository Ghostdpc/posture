import numpy as np
import cv2
import  config
import os
listread=config.datalistpath
#divide video into 70 frames
def framecreate():
  with open(listread,"rb") as fr:
      datalist=fr.read()
  #for eachline in datalist:
  dirlist = []
  filename = []
  for root, child, files in os.walk(config.datafilepath+"/hmdb/"):

          for each in files:
              #create folder for frames
           if not os.path.exists(config.traindatapath+each[0:-4]):
                 os.mkdir(config.traindatapath+each[0:-4])
           cap = cv2.VideoCapture(config.datafilepath+"/hmdb/"+each)
           os.chdir(config.traindatapath+each[0:-4])
           count = 1
           dirlist.append(config.traindatapath+each[0:-4])
           sucess=True
              #read frames from video
           while (sucess):
               sucess, frame = cap.read()
               if sucess==False:
                   print(each[0:-4]+"done")

               cv2.imwrite( str(count) + '.jpg', frame)  # 存储为图像
               filename.append(str(count) + '.jpg')
               count = count + 1

           cap.release()
          #add noise or change light of the video frames
  changelight(dirlist, filename)
  addnoise(dirlist,filename)

def trainlistcreate():
    #create list  of train  filename-numofframes-cls  example: zhengchang0 70 6
    with open("E:/proroot/dataset/test/train_list.txt", "r+") as fr:
        path=config.traindatapath
        for root, child, files in os.walk(path):
            for each in child:
              #print(path+"/"+each)
              temp=os.listdir(path+"/"+each)
              num=len(temp)-1
              if "tuobei" in each:
                  fr.write(each+" "+str(num)+" "+"0"+"\n")
              if "tuosai-zuo" in each:
                  fr.write(each + " " + str(num) + " " + "1" + "\n")
              if "tuosai-you" in each:
                  fr.write(each + " " + str(num) + " " + "2" + "\n")
              if "waitou-zuo" in each:
                  fr.write(each + " " + str(num) + " " + "3" + "\n")
              if "waitou-you" in each:
                  fr.write(each + " " + str(num) + " " + "4" + "\n")
              if "yaobi" in each:
                  fr.write(each + " " + str(num) + " " + "5" + "\n")
              if "zhengchang" in each:
                  fr.write(each + " " + str(num) + " " + "6" + "\n")

def mul():
    dirname=[]
    filename=[]
    for root, child, files in os.walk(config.traindatapath):
        for each in child:
            dirname.append(config.traindatapath+each)
        for a in range(1,72):
            filename.append(str(a)+'.jpg')
    changelight(dirname,filename)
    addnoise(dirname,filename)



def addnoise(dirname,filename):
    for each in dirname:
        lighterdir = each + "noise/"
        if not os.path.exists(lighterdir):
            os.mkdir(lighterdir)
        os.chdir(lighterdir)
        count = 1
        for lines in filename:
            fn = each + "/" + lines
            img = cv2.imread(fn)
            if img is None:
                break
            n = 5000
            for k in range(0, n):
                # get the random point
                xi = int(np.random.uniform(0, img.shape[1]))
                xj = int(np.random.uniform(0, img.shape[0]))
                # add noise
                if img.ndim == 2:
                    img[xj, xi] = 255
                elif img.ndim == 3:
                    img[xj, xi, 0] = 25
                    img[xj, xi, 1] = 20
                    img[xj, xi, 2] = 20
            cv2.imwrite(str(count) + '.jpg', img)
            count = count + 1
def changelight(dirname,filename):
 for each in dirname:
    lighterdir=each+"lighter/"
    if not os.path.exists(lighterdir):
        os.mkdir(lighterdir)
    os.chdir(lighterdir)
    count = 1
    for lines in filename:
        fn=each+"/"+lines
        img = cv2.imread(fn)
        if  img is None:
            break
        w = img.shape[1]
        h = img.shape[0]
        for xi in range(0, w):
          for xj in range(0, h):
            for c in range(0, 3):
                res = int(img[xj, xi, c] * 1.2 + 50)
                if (res < 0):
                    res = 0
                if (res > 255):
                    res = 255
                    img[xj, xi, c] = res
        cv2.imwrite(str(count) + '.jpg',img)
        count = count + 1
    print(each,"lighter done")
    darkerdir = each + "darker/"
    if not os.path.exists(darkerdir):
        os.mkdir(darkerdir)
    os.chdir(darkerdir)
    count = 1
    for lines in filename:
        fn = each + "/" + lines
        img = cv2.imread(fn)
        if img is None:
            break
        w = img.shape[1]
        h = img.shape[0]
        for xi in range(0, w):
            for xj in range(0, h):
                for c in range(0, 3):
                    res = int(img[xj, xi, c] * 0.8 - 50)
                    if (res < 0):
                        res = 0
                    if (res > 255):
                        res = 255
                        img[xj, xi, c] = res
        cv2.imwrite(str(count) + '.jpg', img)
        count = count + 1
    print(each, "darker done")


def changedegree():
    s=3
if __name__ == '__main__':
    print('processing!')
    framecreate()
    #mul()
    #trainlistcreate()