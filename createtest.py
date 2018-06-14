import os
import config
def testlistcreate(listpath):
    #open list path with training path
    with open(listpath, "r+") as fr:
        path=config.testdatapath
        count = 1
        #create test list  file format: path startframe class_id video_id   Example:chengchang1 16 0 6
        for root, child, files in os.walk(path):
            for each in child:
              #print(path+"/"+each)
              temp=os.listdir(path+"/"+each)
              num=len(temp)-1
              total=4
#7 different posture
              if "tuobei" in each:
                  for n in range(0,total):
                    fr.write(each+" "+str(n*16)+" "+"0 "+str(count)+"\n")
              if "tuosai-zuo" in each:
                  for n in range(0, total):
                   fr.write(each + " " + str(n*16) + " " + "1 "+str(count) + "\n")
              if "tuosai-you" in each:
                  for n in range(0, total):
                   fr.write(each + " " + str(n*16) + " " + "2 "+str(count) + "\n")
              if "waitou-zuo" in each:
                  for n in range(0, total):
                   fr.write(each + " " + str(n*16) + " " + "3 " +str(count)+ "\n")
              if "waitou-you" in each:
                  for n in range(0, total):
                   fr.write(each + " " + str(n*16) + " " + "4 "+str(count) + "\n")
              if "yaobi" in each:
                  for n in range(0, total):
                   fr.write(each + " " +str(n*16) + " " + "5 "+str(count) + "\n")
              if "zhengchang" in each:
                  for n in range(0, total):
                   fr.write(each + " " +str(n*16) + " " + "6 "+str(count) + "\n")
              count=count+1
if __name__ == '__main__':
    #create trainlist
    testlistcreate("E:/proroot/dataset/test/test_list.txt")
