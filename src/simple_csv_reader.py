import numpy as np
import matplotlib.pyplot as plt

print 'hello'

raw_data = "data/big_square.csv"

import csv
with open(raw_data) as csvfile:
	reader = csv.DictReader(csvfile)
	print reader.fieldnames
	row_counter = 0
	for row in reader:
		print(row['dt'], row['dq1'])
		row_counter += 1
	print "total rows:", row_counter


#dt,head_des_x,head_des_y,head_des_z,reye_des_x,reye_des_y,reye_des_z,leye_des_x,leye_des_y,leye_des_z,head_cur_x,head_cur_y,head_cur_z,reye_cur_x,reye_cur_y,reye_cur_z,leye_cur_x,leye_cur_y,leye_cur_z,head_ori_error,reye_orientation_error,leye_orientation_error,h1,h2,h3,h4,h5,q0,q1,q2,q3,q4,q5,q6,dq0,dq1,dq2,dq3,dq4,dq5,dq6,rank_t1,rank_t2

