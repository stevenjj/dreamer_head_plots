import numpy as np
import matplotlib.pyplot as plt
import csv


raw_data = "data/big_square.csv"

global datum_timestamp

datum_timestamp = 0.0
t_list = []
head_des_x = []
head_des_y = []
head_des_z = []

reye_des_x = []
reye_des_y = []
reye_des_z = []

head_cur_x = []
head_cur_y = []
head_cur_z = []

reye_cur_x = []
reye_cur_y = []
reye_cur_z = []

def process_row(row):
	global datum_timestamp

	t_list.append(datum_timestamp)
	head_des_x.append(float(row['head_des_x']))
	head_des_y.append(float(row['head_des_y']))
	head_des_z.append(float(row['head_des_z']))	

	head_cur_x.append(float(row['head_cur_x']))
	head_cur_y.append(float(row['head_cur_y']))
	head_cur_z.append(float(row['head_cur_z']))	

	reye_des_x.append(float(row['reye_des_x']))
	reye_des_y.append(float(row['reye_des_y']))
	reye_des_z.append(float(row['reye_des_z']))		

	reye_cur_y.append(float(row['reye_cur_y']))			
	reye_cur_z.append(float(row['reye_cur_z']))


	# Update Time
	datum_timestamp += float(row['dt'])



#dt,head_des_x,head_des_y,head_des_z,reye_des_x,reye_des_y,reye_des_z,leye_des_x,leye_des_y,leye_des_z,head_cur_x,head_cur_y,head_cur_z,reye_cur_x,reye_cur_y,reye_cur_z,leye_cur_x,leye_cur_y,leye_cur_z,head_ori_error,reye_orientation_error,leye_orientation_error,h1,h2,h3,h4,h5,q0,q1,q2,q3,q4,q5,q6,dq0,dq1,dq2,dq3,dq4,dq5,dq6,rank_t1,rank_t2
with open(raw_data) as csvfile:
	reader = csv.DictReader(csvfile)
	print reader.fieldnames
	row_counter = 0
	for row in reader:
		process_row(row)
		row_counter += 1
	print "total rows:", row_counter



t = np.array(t_list)
head_des_y_array = np.array(head_des_y)
head_des_z_array = np.array(head_des_z)

reye_des_y_array = np.array(reye_des_y)
reye_des_z_array = np.array(reye_des_z)


plt.figure(1)
plt.plot(reye_des_y, reye_des_z, marker='o', linestyle='-', markersize=3.0, color='steelblue')
plt.plot(reye_cur_y, reye_cur_z, marker='s', linestyle='--', markersize=2.0, color='lightgreen')

plt.figure(2)
plt.plot(head_des_y, head_des_z, marker='+', linestyle='-', color='red')
plt.plot(head_cur_y, head_cur_z, marker='x', linestyle='--',  markersize=2.0, color='orange')


plt.show()


