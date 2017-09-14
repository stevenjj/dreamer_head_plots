import numpy as np
import matplotlib.pyplot as plt
import csv
#import pandas as pd #this is how I usually import pandas

import matplotlib
#matplotlib.style.use('ggplot')

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

data_name = "within_limits"
#data_name = "no_intermediate_value"
#data_name = "no_joint_limits_big_square"
#data_name = "nice_big_square"
raw_data = "data/" + data_name + ".csv"

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



fig4 = plt.figure(4)
ax41 = plt.subplot2grid((1,1), (0,0))
ax41.plot(head_cur_y, head_cur_z, linestyle='-', color='red', label='Head Y-Z actual',  linewidth=5)
ax41.plot(head_des_y, head_des_z, linestyle='--', color='orange', label='Head Y-Z desired', linewidth=5)	

ax41.plot(reye_cur_y, reye_cur_z, linestyle='-', color='green', marker='x', markersize=10.0, mew=2.0, markevery=10, label='R Eye Y-Z actual',  linewidth=2.5)	
ax41.plot(reye_des_y, reye_des_z, linestyle='--', color='green', marker='o', markersize=10.0, mew=2.0, markevery=10, label='R Eye Y-Z desired', linewidth=2.5)	

ax41.plot(leye_cur_y, leye_cur_z, linestyle='-', color='blue', label='L Eye Y-Z actual',  linewidth=2.5)	
ax41.plot(leye_des_y, leye_des_z, linestyle='--', color='blue', label='L Eye Y-Z desired',  linewidth=2.5)	

ax41.title.set_text('Gaze Trajectory Visualization (Y-Z plane)')
ax41.set_ylabel(r'Z(m)')
ax41.set_xlabel(r'Y(m)')
ax41.legend(['Actual Head Gaze', 'Desired Head Gaze', 'Actual Right Eye Gaze', 'Desired Right Eye Gaze', 'Actual Left Eye Gaze', 'Desired Left Eye Gaze'])
ax41.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax41.set_xlim(-0.6, 0.6)
ax41.set_ylim(-0.6, 0.75)
ax41.set_aspect('equal', adjustable='box')
#ax41.set_aspect('equal', 'datalim')
#ax41.axis('equal')
ax41.grid('on')



fig5 = plt.figure(5)
ax42 = plt.subplot2grid((1,1), (0,0) )
ax42.plot(t, head_cart_error, linestyle='-', color='red', label='H Gaze Error 2-Norm', linewidth=5)
ax42.plot(t, reye_cart_error, linestyle='--', marker='x', markersize=10.0, mew=2.0, markevery=15, color='green', label='R Eye Gaze Error 2-Norm', linewidth=5)
ax42.plot(t, leye_cart_error, linestyle=':', color='blue', label='L Eye Gaze Error 2-Norm', linewidth=5)

ax42.set_ylabel(r'Error $||\cdot||_2$')
ax42.set_xlabel(r'Time')

ax42.grid('on')
ax42.title.set_text('Gaze Cartesian Error (2-Norm)')
#ax21.get_xaxis().set_visible(False)
ax42.legend(['Head Gaze Error', 'Right Eye Gaze Error', 'Left Eye Gaze Error'])
ax42.legend(loc="upper right")

plt.tight_layout()


fig6 = plt.figure(6)

ax43 = plt.subplot2grid((4,1), (0,0))
ax43.plot(t, head_cur_y, linestyle='-', color='red', label='Head Gaze Y', linewidth=2.5)
ax43.plot(t, head_des_y, linestyle='--', color='red', label='Head Des Gaze Y', linewidth=2.5)
ax43.plot(t, reye_cur_y, linestyle='-', color='green', marker='x', markersize=10.0, mew=2.0, markevery=5, label='Right Eye Act Gaze Y',linewidth=2.5)
ax43.plot(t, reye_des_y, linestyle='--', color='green', marker='o', markersize=10.0, mew=2.0, markevery=5, label='Right Eye Des Gaze Y',linewidth=2.5)
ax43.plot(t, leye_cur_y, linestyle='-.', color='blue', label='Left Eye Act Gaze Y',linewidth=2.5)
ax43.plot(t, leye_des_y, linestyle=':', color='blue', label='Left Eye Des Gaze Y',linewidth=2.5)

ax43.title.set_text('Gaze Y Position vs Time')
ax43.set_ylabel(r'Gaze Y Position (m)',fontsize=12)
ax43.set_xlabel(r'Time (s)',fontsize=12)
ax43.grid('on')
ax43.legend(['H_cur gaze y', 'H_des gaze y','R_cur gaze y', 'R_des gaze y', 'Left Act Y', 'Left Des Y'])
ax43.legend(loc="lower left", ncol=2)# ax43.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#        ncol=6, mode="expand", borderaxespad=0.)


ax44 = plt.subplot2grid((4,1), (1,0))
ax44.plot(t, head_cur_z, linestyle='-', color='red', label='Head Gaze Z', linewidth=2.5)
ax44.plot(t, head_des_z, linestyle='--', color='red', label='Head Des Gaze Z', linewidth=2.5)
ax44.plot(t, reye_cur_z, linestyle='-', color='green',  marker='x', markersize=10.0, mew=2.0, markevery=5, label='Right Eye Act Gaze Z',linewidth=2.5)
ax44.plot(t, reye_des_z, linestyle='--', color='green',  marker='o', markersize=10.0, mew=2.0, markevery=5, label='Right Eye Des Gaze Z',linewidth=2.5)
ax44.plot(t, leye_cur_z, linestyle='-.', color='blue', label='Left Eye Act Gaze Z',linewidth=2.5)
ax44.plot(t, leye_des_z, linestyle=':', color='blue', label='Left Eye Des Gaze Z',linewidth=2.5)
ax44.grid('on')
ax44.title.set_text('Gaze Z Position vs Time')
ax44.set_ylabel(r'Gaze Z Position (m)',fontsize=12)
ax44.set_xlabel(r'Time (s)',fontsize=12)
ax44.legend(['H_cur gaze z', 'H_des gaze z', 'R_cur gaze z', 'R_des gaze z'])
# ax44.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#        ncol=6, mode="expand", borderaxespad=0.)
ax44.legend(loc="lower right", ncol=2)


ax45 = plt.subplot2grid((4,1), (2,0))
ax45.plot(t, q4_list, linestyle='-', color='black', label=r'$q_o$ Eye Pitch Joint',linewidth=2.5)
ax45.plot(t, q5_list, linestyle='--', color='green', marker='x', markersize=10.0, mew=2.0, markevery=5, label=r'$q_1$ Right Eye Yaw',linewidth=2.5)
ax45.plot(t, q6_list, linestyle=':', color='blue', label=r'$q_2$ Left Eye Yaw',linewidth=2.5)
ax45.grid('on')
ax45.title.set_text('Eye Joint Positions vs Time')
ax45.set_ylabel(r'Joint Position (rads)',fontsize=12)
ax45.set_xlabel(r'Time (s)',fontsize=12)

ax45.legend(['Eye Pitch', 'R Eye Yaw', 'L Eye Yaw'])
# ax45.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#        ncol=3, mode="expand", borderaxespad=0.)
ax45.legend(loc="upper right")

ax46 = plt.subplot2grid((4,1), (3,0))
ax46.plot(t, h1_list, linestyle='-', color='black', label=r'$h_1$ Eye Pitch Activation Value',linewidth=2.5)
ax46.plot(t, h2_list, linestyle='--', color='green', marker='x', markersize=10.0, mew=2.0, markevery=5, label=r'$h_2$ Right Eye Yaw Activation Value',linewidth=2.5)
ax46.plot(t, h3_list, linestyle=':', color='blue', label=r'$h_3$ Left Eye Yaw Activation Value',linewidth=2.5)
ax46.grid('on')
ax46.legend(['h1', 'h2', 'h3'])

ax46.title.set_text('Task Activation vs Time')
ax46.set_ylabel(r'$h \in [0,1]$',fontsize=12)
ax46.set_xlabel(r'Time (s)',fontsize=12)
ax46.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=3, mode="expand", borderaxespad=0.)
#ax46.legend(loc="upper right")






#plt.tight_layout()



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
fig4.savefig(data_name + '_gaze_yz' + '.png')
fig5.savefig(data_name + '_gaze_cart_error' + '.png')
fig6.savefig(data_name + '_gaze_multiplots' + '.png')


