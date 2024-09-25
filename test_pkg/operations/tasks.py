import numpy as np
from cvxopt import solvers, matrix
import yaml

def go_to_point(robot_pos_old):
    start_positions = robot_pos_old
    # Ddesired positions for robots
    task_positions = np.array([[-4,6,-4,6],
                               [10,10,-12,-12]])
    
    # priorities = np.array([0,1,2,3])
    priorities = np.array([0,1,2,3])
    temp_positions = np.empty(start_positions.shape)
    num_robots = len(start_positions[0])
    num_tasks = len(task_positions[0])
    
    # Simulation parameters
    DT = 0.05
    k = 0.001
    K = 1000
    gam = np.full(num_tasks, 0.1)

    H = np.block([[np.eye(2), np.zeros((2, num_tasks))], [np.zeros((num_tasks, 2)), k * np.eye(num_tasks)]])
    f = np.zeros((2 + num_tasks, 1))

    for i in range(num_robots):
        A = np.zeros((2*num_tasks -1, 2+num_tasks))
        b = np.zeros((2*num_tasks -1, 1))

        a_left_2c = np.zeros((2*num_tasks -1, 2))
        A[0:num_tasks, 2:2+num_tasks] = -1 * np.eye(num_tasks)
        A[num_tasks:2*num_tasks -1,2+priorities[i]] = 1
        less_prio_row_set = num_tasks

        for j in range(num_tasks):
            b[j] = gam[j] * (-np.linalg.norm(start_positions[:,i] - task_positions[:,j]) ** 2)
            a_left_2c[j, 0:2] = 2 * (start_positions[:,i] - task_positions[:,j]).T
            if (less_prio_row_set < 2*num_tasks -1) and (A[less_prio_row_set ,j+2] != 1):
                A[less_prio_row_set, 2+j] = -(1 / K)
                less_prio_row_set = less_prio_row_set + 1

        A[:, 0:2] = a_left_2c
        A[0:num_tasks, 2:2+num_tasks] = -1 * np.eye(num_tasks)
        A[num_tasks:2*num_tasks -1,2+priorities[i]] = 1

        # Solve quadratic programming problems using cvxopt.solve_qp for u
        solvers.options['show_progress'] = False  # Hide progress output
        u = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b))
        v = np.array(u['x'])[0:2]

        temp_positions[0,i] = v[0,0] * DT
        temp_positions[1,i] = v[1,0] * DT

    return temp_positions


def formation_control(robot_pos_old):   
    start_positions = robot_pos_old
    task_position = np.array([[6], [6]])
    temp_positions = np.empty(start_positions.shape)
    # Simulation parameters
    DT = 0.05
    k = 0.001
    K = 1000

    gama = np.array([0.1,0.1])
    # neighbour for a rectrangle
    neighbour = np.array([[0, 4, 3, 5],
                          [4, 0, 5, 3],
                          [3, 5, 0, 4],
                          [5, 3, 4, 0]])
    
    H = np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), k * np.eye(2)]])
    f = np.zeros((4,1))

    Jx = 0
    for i in range(4):
        count1 = 0
        for j in range(4):
            if neighbour[i][j] != 0:
                count1 = count1 + 0.5 * (np.linalg.norm(start_positions[:,i] - start_positions[:,j]) - neighbour[i][j]) ** 2
        Jx = Jx + count1

    dJxi = np.zeros((4,2))
    for i in range(4):
        temp2 = np.zeros((1,2))
        for j in range(4):
            if neighbour[i][j] != 0:
                temp1 = np.linalg.norm(start_positions[:,i] - start_positions[:,j])
                temp2 = temp2 + ((temp1 - neighbour[i][j]) / temp1) * (start_positions[:,i] - start_positions[:,j]).T
        dJxi[i] = 2 * temp2


    A1 = np.block([[dJxi[0], -1, 0], [2 * (start_positions[:,0] - task_position[:,0]).T, 0, -1], [0, 0, 1, -(1 / K)]])
    b1 = np.array([-(gama[0] * Jx), gama[1] * (-np.linalg.norm(start_positions[:,0] - task_position) ** 2), 0])
    b1 = b1.reshape(-1,1)

    A2 = np.block([[dJxi[1], -1, 0], [2 * (start_positions[:,1] - task_position[:,0]).T, 0, -1], [0, 0, -(1 / K), 1]])
    b2 = np.array([-(gama[0] * Jx), gama[1] * (-np.linalg.norm(start_positions[:,1] - task_position) ** 2), 0])
    b2 = b2.reshape(-1,1)

    A3 = np.block([[dJxi[2], -1, 0], [2 * (start_positions[:,2] - task_position[:,0]).T, 0, -1], [0, 0, 1, -(1 / K)]])
    b3 = np.array([-(gama[0] * Jx), gama[1] * (-np.linalg.norm(start_positions[:,2] - task_position) ** 2), 0])
    b3 = b3.reshape(-1,1)

    A4 = np.block([[dJxi[3], -1, 0], [2 * (start_positions[:,3] - task_position[:,0]).T, 0, -1], [0, 0, 1, -(1 / K)]])
    b4 = np.array([-(gama[0] * Jx), gama[1] * (-np.linalg.norm(start_positions[:,3] - task_position) ** 2), 0])
    b4 = b4.reshape(-1,1)

    # Solve quadratic programming problems using cvxopt.solve_qp for u1 and u2
    solvers.options['show_progress'] = False  # Hide progress output
    u1 = solvers.qp(matrix(H), matrix(f), matrix(A1), matrix(b1))
    u2 = solvers.qp(matrix(H), matrix(f), matrix(A2), matrix(b2))
    u3 = solvers.qp(matrix(H), matrix(f), matrix(A3), matrix(b3))
    u4 = solvers.qp(matrix(H), matrix(f), matrix(A4), matrix(b4))

    v1 = np.array(u1['x'])[0:2]
    v2 = np.array(u2['x'])[0:2]
    v3 = np.array(u3['x'])[0:2]
    v4 = np.array(u4['x'])[0:2]

    temp_positions[:,0] = v1[:,0] * DT
    temp_positions[:,1] = v2[:,0] * DT
    temp_positions[:,2] = v3[:,0] * DT
    temp_positions[:,3] = v4[:,0] * DT

    return temp_positions



def hum_rob_int(robot_pos_old):
    start_positions = robot_pos_old
    task_position = np.array([[6], [6]])
    temp_positions = np.empty(start_positions.shape)
    p1= np.array([[0], [10]])
    # Simulation parameters
    DT = 0.05
    k = 0.1
    K = 500
    kh = 1000

    gama = np.array([0.0005,0.1])
    # neighbour for a rectrangle
    neighbour = np.array([[0, 4, 3, 5],
                          [4, 0, 5, 3],
                          [3, 5, 0, 4],
                          [5, 3, 4, 0]])
    
    H = np.block([[100*np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))], [np.zeros((2, 2)), k * np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), np.zeros((2, 2)), kh * np.eye(2)]])
    f = np.zeros((6,1))

    Jx = 0
    for i in range(4):
        count1 = 0
        for j in range(4):
            if neighbour[i][j] != 0:
                count1 = count1 + 0.5 * (np.linalg.norm(start_positions[:,i] - start_positions[:,j]) - neighbour[i][j]) ** 2
        Jx = Jx + count1

    dJxi = np.zeros((4,2))
    for i in range(4):
        temp2 = np.zeros((1,2))
        for j in range(4):
            if neighbour[i][j] != 0:
                temp1 = np.linalg.norm(start_positions[:,i] - start_positions[:,j])
                temp2 = temp2 + ((temp1 - neighbour[i][j]) / temp1) * (start_positions[:,i] - start_positions[:,j]).T
        dJxi[i] = 2 * temp2
    

    A1 = np.block([[dJxi[0], -1, 0, 0, 0], [2 * (start_positions[:,0] - task_position[:,0]).T, 0, -1, 0, 0], [0, 0, 1, -(1 / K), 0, 0]])
    b1 = np.array([-(gama[0] * Jx), gama[1] * (-np.linalg.norm(start_positions[:,0] - task_position) ** 2), 0])
    b1 = b1.reshape(-1,1)

    A2 = np.block([[dJxi[1], -1, 0, 0, 0], [2 * (start_positions[:,1] - task_position[:,0]).T, 0, -1, 0, 0], [0, 0, -(1 / K), 1, 0, 0]])
    b2 = np.array([-(gama[0] * Jx), gama[1] * (-np.linalg.norm(start_positions[:,1] - task_position) ** 2), 0])
    b2 = b2.reshape(-1,1)

    A3 = np.block([[dJxi[2], -1, 0, 0, 0], [2 * (start_positions[:,2] - task_position[:,0]).T, 0, -1, 0, 0], [0, 0, 1, -(1 / K), 0, 0]])
    b3 = np.array([-(gama[0] * Jx), gama[1] * (-np.linalg.norm(start_positions[:,2] - task_position) ** 2), 0])
    b3 = b3.reshape(-1,1)

    A4 = np.block([[dJxi[3], -1, 0, 0, 0], [2 * (start_positions[:,3] - task_position[:,0]).T, 0, -1, 0, 0], [0, 0, 1, -(1 / K), 0, 0]])
    b4 = np.array([-(gama[0] * Jx), gama[1] * (-np.linalg.norm(start_positions[:,3] - task_position) ** 2), 0])
    b4 = b4.reshape(-1,1)
    
    C4 = np.block([np.eye(2), np.zeros((2, 2)), -1 * np.eye(2)])
    #d4 = np.array([6., 4.])
    #d4 = d4.reshape(-1,1)
    d4 = (p1[:,0]- start_positions[:,3])
    d4 = d4.reshape(-1,1)

    # Solve quadratic programming problems using cvxopt.solve_qp for u1 and u2
    solvers.options['show_progress'] = False  # Hide progress output
    u1 = solvers.qp(matrix(H), matrix(f), matrix(A1), matrix(b1))
    u2 = solvers.qp(matrix(H), matrix(f), matrix(A2), matrix(b2))
    u3 = solvers.qp(matrix(H), matrix(f), matrix(A3), matrix(b3))
    u4 = solvers.qp(matrix(H), matrix(f), matrix(A4), matrix(b4), matrix(C4), matrix(d4))

    v1 = np.array(u1['x'])[0:2]
    v2 = np.array(u2['x'])[0:2]
    v3 = np.array(u3['x'])[0:2]
    v4 = np.array(u4['x'])[0:2]

    temp_vec4 = np.array(u4['x'])[4:6]

    temp_positions[:,0] = v1[:,0] * DT
    temp_positions[:,1] = v2[:,0] * DT
    temp_positions[:,2] = v3[:,0] * DT
    temp_positions[:,3] = v4[:,0] * DT

    return temp_positions, temp_vec4





def cent_with_passivity(states_old, haptic):   
    #temp_positions = states_old
    # Simulation parameters
    DT = 0.02

    # Constant parameters
    k = 35.  #kappa
    kh = 1.
    kv = 0.1
    kp = 0.1
    #K = 100

    gama = np.array([0.3,0.1])
    alpha1 = 2.

    Aa = np.block([[np.zeros((2, 2)), np.eye(2)], [np.zeros((2, 2)), np.zeros((2, 2))]])
    Bb = np.block([[np.zeros((2, 2))], [np.eye(2)]])

    pG= np.array([[-0.75], [0.75]])

    neighbour = np.array([[0, 0.8, 0.8, 1.13],
                        [0.8, 0, 1.13, 0.8],
                        [0.8, 1.13, 0, 0.8],
                        [1.13, 0.8, 0.8, 0]])

    
    H = np.block([[np.eye(8), np.zeros((8, 4)), np.zeros((8, 2))], [np.zeros((4, 8)), k * np.eye(4), np.zeros((4, 2))], [np.zeros((2, 8)), np.zeros((2, 4)), kh * np.eye(2)]])
    f = np.zeros((14,1))

    states = states_old
    temp_states = np.empty(states.shape)
    xG = np.zeros((2,1))
    vG = np.zeros((2,1))
    for i in range(4):
        xG = xG + states[0:2,i].reshape(-1,1)
        vG = vG + states[2:4,i].reshape(-1,1)

    xG = (1/4) * xG
    uH = haptic.reshape(-1,1)
    # uH = kp * (pG - xG)
    # uH_bound = 0.1
    # uH[0] = min(uH_bound,max(-uH_bound,uH[0]))
    # uH[1] = min(uH_bound,max(-uH_bound,uH[1]))
    vG = (1/4) * vG


    Jx = np.zeros((4,))
    for i in range(4):
        count1 = 0
        for j in range(4):
            if neighbour[i][j] != 0:
                count1 = count1 + 0.5 * (np.linalg.norm(states[0:2,i] - states[0:2,j]) - neighbour[i][j]) ** 2
        Jx[i] = count1

    Lx = np.zeros((4,))
    for i in range(4):
        Lx[i] = 0.5 * alpha1 * (np.linalg.norm(states[2:4,i])) **2

    V1 = Jx + Lx

    dJxi = np.zeros((4,2))
    for i in range(4):
        temp2 = np.zeros((1,2))
        for j in range(4):
            if neighbour[i][j] != 0:
                temp1 = np.linalg.norm(states[0:2,i] - states[0:2,j])
                temp2 = temp2 + ((temp1 - neighbour[i][j]) / temp1) * (states[0:2,i] - states[0:2,j]).T
        dJxi[i] = 2 * temp2

    dV1 = np.block([dJxi, alpha1 * ((states[2:4]).T)])

    zeros = np.zeros((1,2))
    conv6 = -(uH.T)/4
    eye = (1/4) * np.eye(2)
    A1 = np.block([[dV1[0] @ Bb, zeros, zeros, zeros], [zeros, dV1[1] @ Bb, zeros, zeros], [zeros, zeros, dV1[2] @ Bb, zeros], [zeros, zeros, zeros, dV1[3] @ Bb]])
    A2 = np.block([-1 * np.eye(4)])
    A3 = np.block([[zeros], [zeros], [zeros], [zeros]])
    A4 = np.block([[(states[2:4,0]).T, zeros, zeros, zeros], [zeros, (states[2:4,1]).T, zeros, zeros], [zeros, zeros, (states[2:4,2]).T, zeros], [zeros, zeros, zeros, (states[2:4,3]).T]])
    A5 = np.zeros((4,4))
    A6 = np.block([[conv6], [conv6], [conv6], [conv6]])

    A = np.block([[A1, A2, A3], [A4, A5, A6]])
    b = np.array([-(gama[0] * V1[0]) - (dV1[0] @ Aa @ states[:,0]), -(gama[0] * V1[1]) - (dV1[1] @ Aa @ states[:,1]), -(gama[0] * V1[2]) - (dV1[2] @ Aa @ states[:,2]), -(gama[0] * V1[3]) - (dV1[3] @ Aa @ states[:,3]), 0, 0, 0, 0])
    b = b.reshape(-1,1)
    C = np.block([eye, eye, eye, eye, np.zeros((2, 4)), -kv * np.eye(2)])
    d = kv * uH - kv * vG
    d = d.reshape(-1,1)

    # Solve quadratic programming problems using cvxopt.solve_qp for u
    solvers.options['show_progress'] = False  # Hide progress output
    v = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b), matrix(C), matrix(d))

    u = np.array(v['x'])[0:8]
    delt = np.array(v['x'])[8:12]
    del_uh = np.array(v['x'])[12:14]

    velocity = states[2:4,:].T.flatten() @ u
    velocity_human = (uH.T) @ del_uh
    HI = uH + del_uh

    for i in range(4):
        temp_states[:,i] = (Aa @ states[:,i] + Bb @ u[(i*2):((i*2)+2),0]) * DT
    
    #states = states + temp_states
    #temp_positions = temp_states[0:2]
    vel = temp_states[2:4]

    temp_u_i_values = u
    temp_del_uh_i_values = del_uh
    temp_x2_i_values = states[2:4,:].T.flatten()
    temp_x2_i_values = temp_x2_i_values.reshape(-1,1)


    # Load the array from the YAML file
    with open('/home/rashik/ros2_ws/src/test_pkg/test_pkg/operations/data_with_passivity.yaml', 'r') as file:
        data = yaml.safe_load(file)
        
    t = data['t']
    
    if(t != 0):
        # Read the values from YAML file
        time = np.array(data['time'])
        V1_values = np.array(data['V1_values'])
        xG_values = np.array(data['xG_values'])
        pG_values = np.array(data['pG_values'])
        velocity_values = np.array(data['velocity_values'])
        velocity_human_values = np.array(data['velocity_human_values'])
        u_i_values = np.array(data['u_i_values'])
        del_uh_i_values = np.array(data['del_uh_i_values'])
        uH_values = np.array(data['uH_values'])
        HI_values = np.array(data['HI_values'])
        x2_i_values = np.array(data['x2_i_values'])

        t = t + 1
        time = np.append(time,t)
        V1_values = np.append(V1_values, V1[0]+V1[1]+V1[2]+V1[3])
        xG_values = np.append(xG_values, xG, axis = 1)
        pG_values = np.append(pG_values, pG, axis = 1)
        velocity_values = np.append(velocity_values, velocity)
        velocity_human_values = np.append(velocity_human_values, velocity_human)
        u_i_values = np.append(u_i_values, temp_u_i_values, axis = 1)
        del_uh_i_values = np.append(del_uh_i_values, temp_del_uh_i_values, axis = 1)
        uH_values = np.append(uH_values, uH, axis = 1)
        HI_values = np.append(HI_values, HI, axis = 1)
        x2_i_values = np.append(x2_i_values, temp_x2_i_values, axis = 1)
    else:
        t = t + 1
        time = np.array([t])
        V1_values = np.array(V1[0]+V1[1]+V1[2]+V1[3])
        xG_values = xG
        pG_values = pG
        velocity_values = velocity
        velocity_human_values = velocity_human
        u_i_values = temp_u_i_values
        del_uh_i_values = temp_del_uh_i_values
        uH_values = uH
        HI_values = HI
        x2_i_values = temp_x2_i_values
    

    # Write the updated array back to the YAML file
    data['t'] = t
    data['time'] = time.tolist()
    data['V1_values'] = V1_values.tolist()
    data['xG_values'] = xG_values.tolist()
    data['pG_values'] = pG_values.tolist()
    data['velocity_values'] = velocity_values.tolist()
    data['velocity_human_values'] = velocity_human_values.tolist()
    data['u_i_values'] = u_i_values.tolist()
    data['del_uh_i_values'] = del_uh_i_values.tolist()
    data['uH_values'] = uH_values.tolist()
    data['HI_values'] = HI_values.tolist()
    data['x2_i_values'] = x2_i_values.tolist()


    with open('/home/rashik/ros2_ws/src/test_pkg/test_pkg/operations/data_with_passivity.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)   

    return vel, del_uh




def cent_without_passivity(states_old, haptic):   
    # Simulation parameters
    DT = 0.02

    # Constant parameters
    k = 35.  #kappa
    kh = 1.
    kv = 0.1
    kp = 0.1
    #K = 100

    gama = np.array([0.3,0.1])
    alpha1 = 2.

    Aa = np.block([[np.zeros((2, 2)), np.eye(2)], [np.zeros((2, 2)), np.zeros((2, 2))]])
    Bb = np.block([[np.zeros((2, 2))], [np.eye(2)]])

    pG= np.array([[-0.75], [0.75]])

    neighbour = np.array([[0, 0.8, 0.8, 1.13],
                        [0.8, 0, 1.13, 0.8],
                        [0.8, 1.13, 0, 0.8],
                        [1.13, 0.8, 0.8, 0]])

    
    H = np.block([[np.eye(8), np.zeros((8, 4)), np.zeros((8, 2))], [np.zeros((4, 8)), k * np.eye(4), np.zeros((4, 2))], [np.zeros((2, 8)), np.zeros((2, 4)), kh * np.eye(2)]])
    f = np.zeros((14,1))
    
    states = states_old
    temp_states = np.empty(states.shape)
    xG = np.zeros((2,1))
    vG = np.zeros((2,1))
    for i in range(4):
        xG = xG + states[0:2,i].reshape(-1,1)
        vG = vG + states[2:4,i].reshape(-1,1)

    xG = (1/4) * xG
    uH = haptic.reshape(-1,1)
    # uH = kp * (pG - xG)
    # uH_bound = 0.1
    # uH[0] = min(uH_bound,max(-uH_bound,uH[0]))
    # uH[1] = min(uH_bound,max(-uH_bound,uH[1]))
    vG = (1/4) * vG


    Jx = np.zeros((4,))
    for i in range(4):
        count1 = 0
        for j in range(4):
            if neighbour[i][j] != 0:
                count1 = count1 + 0.5 * (np.linalg.norm(states[0:2,i] - states[0:2,j]) - neighbour[i][j]) ** 2
        Jx[i] = count1

    Lx = np.zeros((4,))
    for i in range(4):
        Lx[i] = 0.5 * alpha1 * (np.linalg.norm(states[2:4,i])) **2

    V1 = Jx + Lx    

    dJxi = np.zeros((4,2))
    for i in range(4):
        temp2 = np.zeros((1,2))
        for j in range(4):
            if neighbour[i][j] != 0:
                temp1 = np.linalg.norm(states[0:2,i] - states[0:2,j])
                temp2 = temp2 + ((temp1 - neighbour[i][j]) / temp1) * (states[0:2,i] - states[0:2,j]).T
        dJxi[i] = 2 * temp2

    dV1 = np.block([dJxi, alpha1 * ((states[2:4]).T)])

    zeros = np.zeros((1,2))
    conv6 = -(uH.T)/4
    eye = (1/4) * np.eye(2)
    A1 = np.block([[dV1[0] @ Bb, zeros, zeros, zeros], [zeros, dV1[1] @ Bb, zeros, zeros], [zeros, zeros, dV1[2] @ Bb, zeros], [zeros, zeros, zeros, dV1[3] @ Bb]])
    A2 = np.block([-1 * np.eye(4)])
    A3 = np.block([[zeros], [zeros], [zeros], [zeros]])

    A = np.block([A1, A2, A3])
    b = np.array([-(gama[0] * V1[0]) - (dV1[0] @ Aa @ states[:,0]), -(gama[0] * V1[1]) - (dV1[1] @ Aa @ states[:,1]), -(gama[0] * V1[2]) - (dV1[2] @ Aa @ states[:,2]), -(gama[0] * V1[3]) - (dV1[3] @ Aa @ states[:,3])])
    b = b.reshape(-1,1)
    C = np.block([eye, eye, eye, eye, np.zeros((2, 4)), -kv * np.eye(2)])
    d = kv * uH - kv * vG
    d = d.reshape(-1,1)

    # Solve quadratic programming problems using cvxopt.solve_qp for u
    solvers.options['show_progress'] = False  # Hide progress output
    v = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b), matrix(C), matrix(d))

    u = np.array(v['x'])[0:8]
    delt = np.array(v['x'])[8:12]
    del_uh = np.array(v['x'])[12:14]

    velocity = states[2:4,:].T.flatten() @ u
    velocity_human = (uH.T) @ del_uh

    for i in range(4):
        temp_states[:,i] = (Aa @ states[:,i] + Bb @ u[(i*2):((i*2)+2),0]) * DT
    
    vel = states[2:4]

    # Load the array from the YAML file
    with open('/home/rashik/ros2_ws/src/test_pkg/test_pkg/operations/data_without_passivity.yaml', 'r') as file:
        data = yaml.safe_load(file)
        
    t = data['t']

    if(t != 0):
        # Read the values from YAML file
        time = np.array(data['time'])
        V1_values = np.array(data['V1_values'])
        xG_values = np.array(data['xG_values'])
        velocity_values = np.array(data['velocity_values'])
        velocity_human_values = np.array(data['velocity_human_values'])

        t = t + 1        
        time = np.append(time,t)
        V1_values = np.append(V1_values, V1[0]+V1[1]+V1[2]+V1[3])
        xG_values = np.append(xG_values, xG, axis = 1)
        velocity_values = np.append(velocity_values, velocity)
        velocity_human_values = np.append(velocity_human_values, velocity_human)
    else:
        t = t + 1
        time = np.array([t])
        V1_values = np.array(V1[0]+V1[1]+V1[2]+V1[3])
        xG_values = xG
        velocity_values = velocity
        velocity_human_values = velocity_human

    # Write the updated array back to the YAML file
    data['t'] = t
    data['time'] = time.tolist()
    data['V1_values'] = V1_values.tolist()
    data['xG_values'] = xG_values.tolist()
    data['velocity_values'] = velocity_values.tolist()
    data['velocity_human_values'] = velocity_human_values.tolist()

    # Write the updated array back to the YAML file
    data['my_array'] = vel.tolist()  # Convert numpy array back to list

    with open('/home/rashik/ros2_ws/src/test_pkg/test_pkg/operations/data_without_passivity.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)   

    return vel, del_uh