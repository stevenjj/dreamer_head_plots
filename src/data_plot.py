import numpy as np
import matplotlib.pyplot as plt
import csv
#import pandas as pd #this is how I usually import pandas

import matplotlib
#matplotlib.style.use('ggplot')



#from matplotlib.font_manager import FontProperties

#fontP = FontProperties()
#fontP.set_size('small')
#fontP.set_size(12)

from matplotlib import rc

matplotlib.rcParams.update({'font.size': 12})

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)




#data_name = "within_limits"
#data_name = "no_intermediate_value"
#data_name = "no_joint_limits_big_square"
#data_name = "nice_big_square"
#data_name = "fixed_point_no_intermediate_task"
data_name = "fixed_point_nice_square"
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



fig4 = plt.figure(4, figsize=(5.5, 5.5))
ax41 = plt.subplot2grid((1,1), (0,0))
ax41.plot(head_cur_y, head_cur_z, linestyle='-', color='red', label='Head Y-Z actual',  linewidth=1.5)
ax41.plot(reye_cur_y, reye_cur_z, linestyle='-', color='green', marker='o', markersize=5.0, mew=1.0, markevery=10, label='Right Eye Y-Z actual',  linewidth=1.5)	


ax41.plot(leye_cur_y, leye_cur_z, linestyle='-', color='blue', label='Left Eye Y-Z actual',  linewidth=1.5)	
ax41.plot(head_des_y, head_des_z, linestyle='--', color='orange', label='Head Y-Z desired', linewidth=2.5)	

ax41.plot(reye_des_y, reye_des_z, linestyle='--', color='green', marker='x', markersize=5.0, mew=1.0, markevery=10, label='Right Eye Y-Z desired', linewidth=1.5)	
ax41.plot(leye_des_y, leye_des_z, linestyle='--', color='blue', label='Left Eye Y-Z desired',  linewidth=1.5)	

ax41.title.set_text(r'Gaze Trajectory Visualization (Y-Z plane)')
ax41.set_ylabel(r'Z(m)')
ax41.set_xlabel(r'Y(m)')
ax41.legend([r'Actual Head Gaze', r'Desired Head Gaze', r'Actual Right Eye Gaze', r'Desired Right Eye Gaze', r'Actual Left Eye Gaze', r'Desired Left Eye Gaze'])
ax41.legend(loc="lower center",ncol=2, prop={'size': 12})
#ax41.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax41.set_xlim(-0.7, 0.7)
ax41.set_ylim(-0.7, 0.7)
ax41.set_aspect('equal', adjustable='box')
#ax41.set_aspect('equal', 'datalim')
#ax41.axis('equal')
#ax41.grid('on')	

for item in ([ax41.title, ax41.xaxis.label, ax41.yaxis.label] +
             ax41.get_xticklabels() + ax41.get_yticklabels()):
    item.set_fontsize(12)

plt.tight_layout()

fig5 = plt.figure(5, figsize=(5.5, 3.0))
ax42 = plt.subplot2grid((1,1), (0,0) )
ax42.plot(t, head_cart_error, linestyle='-', color='red', label='Head', linewidth=1.5)
ax42.plot(t, reye_cart_error, linestyle='--', marker='o', markersize=5.0, mew=1.0, markevery=10, color='green', label='Right Eye', linewidth=1.5)
ax42.plot(t, leye_cart_error, linestyle='-', color='blue', label='Left Eye', linewidth=1.5)

ax42.set_ylabel('Gaze Cartesian\nError $||\cdot||_2$', multialignment='center')
ax42.set_ylim(-0.1, 0.4)
ax42.set_xlabel(r'Time')

#ax42.title.set_text(r'Gaze Cartesian Error (2-Norm)')
ax42.legend([r'Head Gaze Error', r'Right Eye Gaze Error', r'Left Eye Gaze Error'])
ax42.legend(loc="lower center",ncol=3, prop={'size': 12})

#box = ax42.get_position()
#ax42.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax42.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()


fig6 = plt.figure(6, figsize=(11.5, 3.5))

ax43 = plt.subplot2grid((2,2), (0,0))
ax43.plot(t, head_cur_y, linestyle='-', color='red', label='Head Gaze Y', linewidth=1.5)
ax43.plot(t, head_des_y, linestyle='--', color='red', label='Head Des Gaze Y', linewidth=1.5)

ax43.title.set_text(r'Head Gaze Positions')
ax43.set_ylabel('Y\nPosition (m)', multialignment='center')
ax43.set_ylim(-0.8, 0.8)
#ax43.legend(['H_cur gaze y', 'H_des gaze y','R_cur gaze y', 'R_des gaze y', 'Left Act Y', 'Left Des Y'])
#ax43.legend(loc="lower left", prop={'size': 12})

ax432 = plt.subplot2grid((2,2), (0,1))
ax432.plot(t, reye_cur_y, linestyle='-', color='green', marker='o', markersize=5.0, mew=1.0, markevery=10, label='Right Eye Act Gaze Y',linewidth=1.5)
ax432.plot(t, reye_des_y, linestyle='--', color='green', marker='x', markersize=5.0, mew=1.0, markevery=10, label='Right Eye Des Gaze Y',linewidth=1.5)
ax432.plot(t, leye_cur_y, linestyle='-', color='blue', label='Left Eye Act Gaze Y',linewidth=1.5)
ax432.plot(t, leye_des_y, linestyle='--', color='blue', label='Left Eye Des Gaze Y',linewidth=1.5)

ax432.title.set_text(r'Eye Gaze Positions')
#ax432.set_ylabel('Gaze Y\nPosition (m)', multialignment='center')
ax432.set_ylim(-0.3, 0.3)

ax44 = plt.subplot2grid((2,2), (1,0))
ax44.plot(t, head_cur_z, linestyle='-', color='red', label='Actual Head Gaze', linewidth=1.5)
ax44.plot(t, head_des_z, linestyle='--', color='red', label='Desired Head Gaze', linewidth=1.5)

# ax44.title.set_text(r'Head Gaze Z Position')
ax44.set_ylabel('Z\nPosition (m)', multialignment='center')
ax44.set_ylim(-0.8, 0.8)
ax44.legend(['H_cur gaze z', 'H_des gaze z', 'R_cur gaze z', 'R_des gaze z'])
ax44.legend(loc="lower left", ncol=2, prop={'size': 12})
ax44.set_xlabel(r'Time (s)',fontsize=12)
#box = ax44.get_position()
# ax44.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax44.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

ax442 = plt.subplot2grid((2,2), (1,1))
ax442.plot(t, reye_cur_z, linestyle='-', color='green',  marker='o', markersize=5.0, mew=1.0, markevery=10, label='Right Eye Actual',linewidth=1.5)
ax442.plot(t, reye_des_z, linestyle='--', color='green',  marker='x', markersize=5.0, mew=1.0, markevery=10, label='Right Eye Desired',linewidth=1.5)
ax442.plot(t, leye_cur_z, linestyle='-', color='blue', label='Left Eye Actual',linewidth=1.5)
ax442.plot(t, leye_des_z, linestyle='--', color='blue', label='Left Eye Desired',linewidth=1.5)

ax442.set_xlabel(r'Time (s)',fontsize=12)
#plt.title(r'Eye Gaze Position')
#ax442.set_ylabel('Z\nPosition (m)', multialignment='center')
ax442.set_ylim(-0.3, 0.3)
ax442.legend(loc="lower center", ncol=2, prop={'size': 12})

# box = ax442.get_position()
# ax442.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax442.legend(loc='center left', bbox_to_anchor=(1, 0.5))



fig7 = plt.figure(7, figsize=(11.5, 1.75))
ax45 = plt.subplot2grid((1,1), (0,0), colspan=1)
ax45.plot(t, q4_list, linestyle='-', color='black', marker='s', markersize=5.0, mew=1.0, markevery=10, label=r'$q_4$ Eye Pitch',linewidth=1.5)
ax45.plot(t, q5_list, linestyle='--', color='green', marker='o', markersize=5.0, mew=1.0, markevery=10, label=r'$q_5$ R Eye Yaw',linewidth=1.5)
ax45.plot(t, q6_list, linestyle='-', color='blue', label=r'$q_6$ L Eye Yaw',linewidth=1.5)
#ax45.title.set_text(r'Eye Joint Positions')
ax45.set_ylabel('Eye Joint\nPosition (rads)', multialignment='center')
ax45.set_xlabel(r'Time (s)')
#ax45.get_xaxis().set_visible(False)
ax45.set_ylim(-0.7, 0.7)
ax45.legend(loc='upper right', prop={'size': 12})
#ax45.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#       ncol=3, mode="expand", borderaxespad=0., prop={'size':12})

#ax45.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

plt.tight_layout()

fig8 = plt.figure(8, figsize=(11.5, 1.75))
ax46 = plt.subplot2grid((1,1), (0,0))
ax46.plot(t, h1_list, linestyle='-', color='black',  marker='s', markersize=5.0, mew=1.0, markevery=10, label=r'$h_4$ Eye Pitch',linewidth=1.5)
ax46.plot(t, h2_list, linestyle='--', color='green', marker='o', markersize=5.0, mew=1.0, markevery=10, label=r'$h_5$ R Eye Yaw',linewidth=1.5)
ax46.plot(t, h3_list, linestyle='-', color='blue', label=r'$h_6$ L Eye Yaw',linewidth=1.5)
ax46.legend(['h4', 'h5', 'h6'])

#ax46.title.set_text(r'Task Activation')
ax46.set_ylim(0.0, 1.1)
ax46.set_ylabel('Task Activation\n $h \in [0,1]$', multialignment='center')
ax46.set_xlabel(r'Time (s)')
ax46.legend(loc="upper right", prop={'size': 12})
# ax46.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#       ncol=3, mode="expand", borderaxespad=0., prop={'size':12})
#ax46.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)

#ax46.legend(loc="upper right")
# box = ax46.get_position()
# ax46.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax46.legend(loc='center left', bbox_to_anchor=(1, 0.5))


plt.tight_layout()

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
fig4.savefig(data_name + '_gaze_yz' + '.pdf')
fig5.savefig(data_name + '_gaze_cart_error' + '.pdf')
fig6.savefig(data_name + '_gaze_multiplots' + '.pdf')
fig7.savefig(data_name + '_gaze_joints' + '.pdf')
fig8.savefig(data_name + '_gaze_activations' + '.pdf')
