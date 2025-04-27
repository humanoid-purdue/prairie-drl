import matplotlib.pyplot as plt
import numpy as np

def fwd_back():
    traj_back = np.genfromtxt("centroid_traj_back.csv", delimiter =',')
    traj_fwd = np.genfromtxt("centroid_traj_fwd.csv", delimiter = ',')
    x = np.arange(10000) / 1000
    plt.plot(x, traj_back[:, 1], label = "backwards walk v_x_des = -0.2")
    plt.axhline(y=-0.2, color='b', linestyle='--', label = "target backwards velocity")
    plt.plot(x, traj_fwd[:, 1], label = "fwd walk v_x_des = 0.2")
    plt.axhline(y=0.2, color='r', linestyle='--', label = "target forwards velocity")
    plt.legend()
    plt.xlabel("time elapsed (s)")
    plt.ylabel("Pelvis forward velocity (m/s)")
    plt.title("Graph of target vs actual forward and backwards base velocity")
    plt.show()

def side():
    traj_side = np.genfromtxt("centroid_traj_side.csv", delimiter=',')
    traj_fwd = np.genfromtxt("centroid_traj_fwd.csv", delimiter=',')
    x = np.arange(10000) / 1000
    plt.plot(x, traj_side[:, 2], label="Sideways walk v_y_des = 0.2")
    plt.axhline(y=0.2, color='b', linestyle='--', label="target sidestep velocity")
    plt.plot(x, traj_fwd[:, 2], label="No Sideways walk v_y_des = 0.0")
    plt.axhline(y=0.0, color='r', linestyle='--', label="target 0 side velocity")
    plt.legend()
    plt.xlabel("time elapsed (s)")
    plt.ylabel("Pelvis sideways velocity (m/s)")
    plt.title("Graph of target vs actual sidestepping base velocity")
    plt.show()

def turning():
    traj_angvel = np.genfromtxt("centroid_traj_angvel.csv", delimiter=',')
    traj_fwd = np.genfromtxt("centroid_traj_fwd.csv", delimiter=',')
    x = np.arange(10000) / 1000
    plt.plot(x, traj_angvel[:, 3], label="Turning angvel_z = -0.7")
    plt.axhline(y=-0.7, color='b', linestyle='--', label="target turning angvel")
    plt.plot(x, traj_fwd[:, 3], label="No Turning angvel_z = 0.0")
    plt.axhline(y=0.0, color='r', linestyle='--', label="target 0 angvel")
    plt.legend()
    plt.xlabel("time elapsed (s)")
    plt.ylabel("Pelvis z angular velocity (rad/s)")
    plt.title("Graph of target vs actual base z angular velocity")
    plt.show()
#side()
turning()