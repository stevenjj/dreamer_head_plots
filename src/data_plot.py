import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd #this is how I usually import pandas

import matplotlib
#matplotlib.style.use('ggplot')

raw_data = "data/nice_big_square.csv"

global datum_timestamp

datum_timestamp = 0.0
t_list = []
head_des_x = []
head_des_y = []
head_des_z = []

leye_des_x = []
leye_des_y = []
leye_des_z = []

reye_des_x = []
reye_des_y = []
reye_des_z = []

head_cur_x = []
head_cur_y = []
head_cur_z = []

leye_cur_x = []
leye_cur_y = []
leye_cur_z = []

reye_cur_x = []
reye_cur_y = []
reye_cur_z = []

head_cart_error = []
reye_cart_error = []
leye_cart_error = []

q0_list = []
q1_list = []
q2_list = []
q3_list = []
q4_list = []
q5_list = []
q6_list = []

h1_list = []
h2_list = []
h3_list = []

def calc_error(x,y,z, x_des, y_des, z_des):
	error_vec = np.array([x_des - x, y_des - y, z_des - z])
	return np.linalg.norm(error_vec)

def process_row(row):
	global datum_timestamp

	t_list.append(datum_timestamp)

	# Process Head Gaze
	h_des_x, h_des_y, h_des_z = float(row['head_des_x']), float(row['head_des_y']), float(row['head_des_z'])
	h_cur_x, h_cur_y, h_cur_z = float(row['head_cur_x']), float(row['head_cur_y']), float(row['head_cur_z'])
	h_error = calc_error(h_cur_x, h_cur_y, h_cur_z, h_des_x, h_des_y, h_des_z)

	head_des_x.append(h_des_x)
	head_des_y.append(h_des_y)
	head_des_z.append(h_des_z)	

	head_cur_x.append(h_cur_x)
	head_cur_y.append(h_cur_y)
	head_cur_z.append(h_cur_z)	
	head_cart_error.append(h_error)

	# Process Right Eye Gaze
	re_des_x, re_des_y, re_des_z = float(row['reye_des_x']), float(row['reye_des_y']), float(row['reye_des_z'])
	re_cur_x, re_cur_y, re_cur_z = float(row['reye_cur_x']), float(row['reye_cur_y']), float(row['reye_cur_z'])
	re_error = calc_error(re_cur_x, re_cur_y, re_cur_z, re_des_x, re_des_y, re_des_z)

	reye_des_x.append(re_des_x)
	reye_des_y.append(re_des_y)	
	reye_des_z.append(re_des_z)		

	reye_cur_x.append(re_cur_x)
	reye_cur_y.append(re_cur_y)
	reye_cur_z.append(re_cur_z)
	reye_cart_error.append(re_error)

	# Process Left Eye Gaze
	le_des_x, le_des_y, le_des_z = float(row['leye_des_x']), float(row['leye_des_y']), float(row['leye_des_z'])
	le_cur_x, le_cur_y, le_cur_z = float(row['leye_cur_x']), float(row['leye_cur_y']), float(row['leye_cur_z'])
	le_error = calc_error(le_cur_x, le_cur_y, le_cur_z, le_des_x, le_des_y, le_des_z)

	leye_des_x.append(le_des_x)
	leye_des_y.append(le_des_y)	
	leye_des_z.append(le_des_z)		

	leye_cur_x.append(le_cur_x)
	leye_cur_y.append(le_cur_y)	
	leye_cur_z.append(le_cur_z)			

	leye_cart_error.append(le_error)


	q4_list.append(float(row['q4']))
	q5_list.append(float(row['q5']))	
	q6_list.append(float(row['q6']))		

	h1_list.append(float(row['h1']))
	h2_list.append(float(row['h2']))	
	h3_list.append(float(row['h3']))		

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


fig = plt.figure(1)	

ax1 = fig.add_subplot(2,2, 1)
ax1.plot(t, head_cur_y, linestyle='-', color='red', label='Head Gaze Y')
ax1.plot(t, head_des_y, linestyle='--', color='red', label='Head Des Gaze Y')
ax1.grid('on')
ax1.legend(['H_cur gaze y', 'H_des gaze y'])
ax1.legend(loc="lower right")

ax2 = fig.add_subplot(2,2, 2)
ax2.plot(t, head_cur_z, linestyle='-', color='red', label='Head Gaze Z')
ax2.plot(t, head_des_z, linestyle='--', color='red', label='Head Des Gaze Z')
ax2.grid('on')
ax2.legend(['H_cur gaze z', 'H_des gaze z'])
ax2.legend(loc="lower right")

ax3 = fig.add_subplot(2,2, 3)
ax3.plot(t, reye_cur_y, linestyle='-', color='green', marker='x', markersize=5.0, markevery=10, label='Right Eye Act Gaze Y')
ax3.plot(t, reye_des_y, linestyle='--', color='green', marker='x', markersize=5.0, markevery=10, label='Right Eye Des Gaze Y')
ax3.plot(t, leye_cur_y, linestyle='-.', color='blue', label='Left Eye Act Gaze Y')
ax3.plot(t, leye_des_y, linestyle=':', color='blue', label='Left Eye Des Gaze Y')
ax3.grid('on')
ax3.legend(['R_cur gaze y', 'R_des gaze y', 'Left Act Y', 'Left Des Y'])
ax3.legend(loc="lower right")

ax4 = fig.add_subplot(2,2, 4)
ax4.plot(t, reye_cur_z, linestyle='-', color='green', marker='x', markersize=5.0, markevery=10, label='Right Eye Act Gaze Z')
ax4.plot(t, reye_des_z, linestyle='--', color='green', marker='x', markersize=5.0, markevery=10, label='Right Eye Des Gaze Z')
ax4.plot(t, leye_cur_z, linestyle='-.', color='blue', label='Left Eye Act Gaze Z')
ax4.plot(t, leye_des_z, linestyle=':', color='blue', label='Left Eye Des Gaze Z')
ax4.grid('on')
ax4.legend(['R_cur gaze z', 'R_des gaze z'])
ax4.legend(loc="lower right")


fig2 = plt.figure(2)	
ax21 = fig2.add_subplot(3,1, 1)
ax21.plot(t, head_cart_error, linestyle='-', color='red', label='H Gaze Error 2-Norm')
ax21.plot(t, reye_cart_error, linestyle='--', marker='x', markersize=5.0, markevery=10, color='green', label='R Eye Gaze Error 2-Norm')
ax21.plot(t, leye_cart_error, linestyle=':', color='blue', label='L Eye Gaze Error 2-Norm')
ax21.grid('on')
ax21.title.set_text('Gaze Cartesian Error (2-Norm)')
ax21.get_xaxis().set_visible(False)
ax21.legend(['Head Gaze Error', 'Right Eye Gaze Error', 'Left Eye Gaze Error'])
ax21.legend(loc="upper left")

ax22 = fig2.add_subplot(3,1,2)
ax22.plot(t, q4_list, linestyle='-', color='black', label='Eye Pitch')
ax22.plot(t, q5_list, linestyle='--', color='green', label='R Eye Yaw')
ax22.plot(t, q6_list, linestyle=':', color='blue', label='L Eye Yaw')
ax22.grid('on')
ax22.title.set_text('Eye Joint Positions')
ax22.legend(['Eye Pitch', 'R Eye Yaw', 'L Eye Yaw'])
ax22.legend(loc="upper left")

ax23 = fig2.add_subplot(3,1,3)
ax23.plot(t, h1_list, linestyle='-', color='black', label='Eye Pitch')
ax23.plot(t, h2_list, linestyle='--', color='green', label='R Eye Yaw')
ax23.plot(t, h3_list, linestyle=':', color='blue', label='L Eye Yaw')
ax23.legend(['h1', 'h2', 'h3'])
ax23.legend(loc="upper left")



fig3 = plt.figure(3)
ax31 = fig3.add_subplot(1,1,1)
ax31.plot(head_cur_y, head_cur_z, linestyle='-', color='red', label='Head Y-Z actual')
ax31.plot(head_des_y, head_des_z, linestyle='--', color='orange', label='Head Y-Z desired')	

ax31.plot(reye_cur_y, reye_cur_z, linestyle='-', color='green', label='R Eye Y-Z actual')	
ax31.plot(reye_des_y, reye_des_z, linestyle='--', color='green', label='R Eye Y-Z actual')	

ax31.plot(leye_cur_y, leye_cur_z, linestyle='-', color='blue', label='L Eye Y-Z actual')	
ax31.plot(leye_des_y, leye_des_z, linestyle='--', color='blue', label='L Eye Y-Z actual')	

#ax23 = fig2.add_subplot(3,1,3)
#df2 = pd.DataFrame( np.zeros( (3,len(t)) ) )
#df2.plot.barh(ax=ax23, stacked=True);


# ax2.plot(t, reye_cur_y, marker='o', markersize=0.1, linestyle='-', color='green', label='Right Eye Gaze Z')
# ax2.plot(t, reye_des_y, marker='s', markersize=0.3, linestyle='--', color='green', label='Right Eye Des Gaze Z')
# ax2.plot(t, leye_cur_y, marker='o', markersize=0.1, linestyle='-', color='blue', label='Blue Eye Gaze Y')
# ax2.plot(t, leye_des_y, marker='s', markersize=0.3, linestyle='--', color='blue', label='Blue Eye Des Gaze Y')

# df2 = pd.DataFrame(np.random.rand(10, 3), columns=['h1', 'h2', 'h3'])
# df2.plot.barh(stacked=True);



plt.show()


