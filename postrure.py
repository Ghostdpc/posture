import tensorflow as tf
from scipy.io import loadmat

traindata=loadmat("D:\\baiduyun\\HMDB_a large human motion database\\50_FIRST_DATES_sit_f_cm_np1_fr_med_24.avi.tform.mat")
print(traindata['BB'])
