import os
path="E:/data2/zhengchang/"
name="zhengchang"
for root, child, files in os.walk(path):
    i=1
    os.chdir(path)
    for each in files:
        os.rename(each, name+str(i)+".avi")
        i=i+1