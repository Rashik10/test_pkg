import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist, PoseStamped, Vector3
from rclpy.qos import ReliabilityPolicy, QoSProfile
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from tf_transformations import euler_from_quaternion
from .operations import tasks
import numpy as np

num_robot = 4

class MyNode(Node):
    def __init__(self):
        # call super() in the constructor to initialize the Node object
        # the parameter we pass is the node name
        super().__init__('test_node')

        # create the publisher object
        self.publisher_1 = self.create_publisher(Twist, 'Mavic_2_PRO_1/cmd_vel', 1)
        self.publisher_2 = self.create_publisher(Twist, 'Mavic_2_PRO_2/cmd_vel', 1)
        self.publisher_3 = self.create_publisher(Twist, 'Mavic_2_PRO_3/cmd_vel', 1)
        self.publisher_4 = self.create_publisher(Twist, 'Mavic_2_PRO_4/cmd_vel', 1)
        self.publisher_haptic = self.create_publisher(Float64MultiArray, '/fd/fd_controller/commands', 1)
        
        # create the subscriber object
        self.subscriber_1 = self.create_subscription(
            PointStamped,
            '/Mavic_2_PRO_1/gps',
            self.listener_callback_1,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.subscriber_2 = self.create_subscription(
            PointStamped,
            '/Mavic_2_PRO_2/gps',
            self.listener_callback_2,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.subscriber_3 = self.create_subscription(
            PointStamped,
            '/Mavic_2_PRO_3/gps',
            self.listener_callback_3,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.subscriber_4 = self.create_subscription(
            PointStamped,
            '/Mavic_2_PRO_4/gps',
            self.listener_callback_4,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        # Subscribe to the /gps/speed_vector topic
        self.speed_vector_subscription_1 = self.create_subscription(Vector3, '/Mavic_2_PRO_1/gps/speed_vector', self.speed_vector_callback_1, 1)
        self.speed_vector_subscription_2 = self.create_subscription(Vector3, '/Mavic_2_PRO_2/gps/speed_vector', self.speed_vector_callback_2, 1)
        self.speed_vector_subscription_3 = self.create_subscription(Vector3, '/Mavic_2_PRO_3/gps/speed_vector', self.speed_vector_callback_3, 1)
        self.speed_vector_subscription_4 = self.create_subscription(Vector3, '/Mavic_2_PRO_4/gps/speed_vector', self.speed_vector_callback_4, 1)
        
        self.imu_subscription_1 = self.create_subscription(Imu, '/Mavic_2_PRO_1/imu', self.imu_callback_1, 1)
        self.imu_subscription_2 = self.create_subscription(Imu, '/Mavic_2_PRO_2/imu', self.imu_callback_2, 1)
        self.imu_subscription_3 = self.create_subscription(Imu, '/Mavic_2_PRO_3/imu', self.imu_callback_3, 1)
        self.imu_subscription_4 = self.create_subscription(Imu, '/Mavic_2_PRO_4/imu', self.imu_callback_4, 1)
        self.ee_pose_subscription = self.create_subscription(PoseStamped, '/fd/ee_pose', self.pose_callback, 1)

        # define the variable to save the received info
        self.x_val = np.empty(num_robot)
        self.y_val = np.empty(num_robot)
        self.x_val_velocity = np.empty(num_robot)
        self.y_val_velocity = np.empty(num_robot)
        self.haptic_position = np.empty(2)
        self.roll = np.empty(num_robot)
        self.pitch = np.empty(num_robot)
        self.yaw = np.empty(num_robot)
        # self.robot_pos_old = np.empty((2,num_robot))
        self.states_old = np.empty((4,num_robot))
        self.feedback = Float64MultiArray()
        # create a Twist message
        self.cmd = []
        for i in range(num_robot):
            self.cmd.append(Twist())
        self.timer = self.create_timer(0.1, self.motion)
        
    def listener_callback_1(self, msg):
        self.x_val[0] = msg.point.x
        self.y_val[0] = msg.point.y
        # print the log info in the terminal
        #self.get_logger().info('I receive: "%s"' % str(self.x_val))
        #self.get_logger().info("x: {}       y: {}".format(self.x_val,self.y_val))
    def listener_callback_2(self, msg):
        self.x_val[1] = msg.point.x
        self.y_val[1] = msg.point.y
    def listener_callback_3(self, msg):
        self.x_val[2] = msg.point.x
        self.y_val[2] = msg.point.y
    def listener_callback_4(self, msg):
        self.x_val[3] = msg.point.x
        self.y_val[3] = msg.point.y

    def speed_vector_callback_1(self, msg):
        self.x_val_velocity[0] = msg.x
        self.y_val_velocity[0] = msg.y
    def speed_vector_callback_2(self, msg):
        self.x_val_velocity[1] = msg.x
        self.y_val_velocity[1] = msg.y
    def speed_vector_callback_3(self, msg):
        self.x_val_velocity[2] = msg.x
        self.y_val_velocity[2] = msg.y
    def speed_vector_callback_4(self, msg):
        self.x_val_velocity[3] = msg.x
        self.y_val_velocity[3] = msg.y    
    
    def imu_callback_1(self, msg):
        # Extract quaternion data from the IMU message
        orientation = msg.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        # Convert quaternion to Euler angles
        self.roll[0], self.pitch[0], self.yaw[0] = euler_from_quaternion(quaternion)
        ##self.get_logger().info("Yaw angle: {}".format(self.yaw))
    def imu_callback_2(self, msg):
        orientation = msg.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.roll[1], self.pitch[1], self.yaw[1] = euler_from_quaternion(quaternion)
    def imu_callback_3(self, msg):
        orientation = msg.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.roll[2], self.pitch[2], self.yaw[2] = euler_from_quaternion(quaternion)
    def imu_callback_4(self, msg):
        orientation = msg.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.roll[3], self.pitch[3], self.yaw[3] = euler_from_quaternion(quaternion)
    
    def pose_callback(self, msg):
        # Read the linear x and y positions from the PoseStamped message
        self.haptic_position[0] = msg.pose.position.x
        self.haptic_position[1] = msg.pose.position.y
        self.get_logger().info("x: {}       y: {}".format(self.haptic_position[0],self.haptic_position[1]))

    def motion(self):
        # print the data
        # self.get_logger().info('I receive: "%s"' % str(self.laser_forward))
        for i in range(num_robot):
            # self.robot_pos_old[0,i] = self.x_val[i]
            # self.robot_pos_old[1,i] = self.y_val[i]
            self.states_old[0,i] = self.x_val[i]
            self.states_old[1,i] = self.y_val[i]
            self.states_old[2,i] = self.x_val_velocity[i]
            self.states_old[3,i] = self.y_val_velocity[i]

        # Logic of move
        #self.robot_pos_new = tasks.formation_control(self.robot_pos_old)          #use when go_to_point or formation_control       
        #self.robot_pos_new, force = tasks.hum_rob_int(self.robot_pos_old)           #use when hum_rob_int used
        # self.robot_pos_new, force = tasks.cent_with_passivity(self.robot_pos_old, self.haptic_position)         #when haptic feedback added
        self.vel, force = tasks.cent_with_passivity(self.states_old, self.haptic_position)
        
        for i in range(num_robot):
            self.cmd[i].linear.x = self.vel[0,i] * 100.5             #go_to_point = speed * 10
            self.cmd[i].linear.y = self.vel[1,i] * 100.5             #formation_control = speed * 1
            self.cmd[i].angular.x = -self.roll[i]
            self.cmd[i].angular.y = -self.pitch[i]
            self.cmd[i].angular.z = -self.yaw[i]                            #hum_rob_int = = speed * 0.5
            
        self.feedback.data = [force[0,0]*200, force[1,0]*200, 0.0]
        
        # self.get_logger().info("x: {}       y: {}".format(force[0,0]*200,force[1,0]*200))   #use when hum_rob_int used
        # Publishing the cmd_vel values to a Topic
        self.publisher_1.publish(self.cmd[0])
        self.publisher_2.publish(self.cmd[1])
        self.publisher_3.publish(self.cmd[2])
        self.publisher_4.publish(self.cmd[3])
        self.publisher_haptic.publish(self.feedback)
        

        # Display the message on the console
        #self.get_logger().info('Publishing: "%s"' % self.cmd)



def main(args=None):
    # initialize the ROS2 communication
    rclpy.init(args=args)
    # declare the node constructor
    node = MyNode()
    # keeps the node alive, waits for a request to kill the node (ctrl+c)
    rclpy.spin(node)
    # Explicity destroy the node
    node.destroy_node()
    # shutdown the ROS2 communication
    rclpy.shutdown()

if __name__ == '__main__':
    main()