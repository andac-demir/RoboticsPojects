#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import serial
# to convert roll, pitch. yaw to quaterions
import tf
# to publish imu data and magnetometer data
from sensor_msgs.msg import Imu, MagneticField
import numpy as np


def _get_topics(node):
    '''
    Description
    imu node is publishing to topics using message type Float64
    queue_size limits amount of queued messages if any subscriber is not
    receiving them fast enough
    Parameters:
        node (str)
    Return:
        pubs (list of <rospy.Publisher>)
    '''
    imu_pub = rospy.Publisher(node, Imu, queue_size=3)
    magneto_pub = rospy.Publisher(node + "/magnetic", MagneticField, queue_size=3)
    return imu_pub, magneto_pub


def talker():
    '''
    Description
    Node publishes sensor measurements to different topics
    '''
    SENSOR_NAME = "imu"
    # This tells rospy the name of your node: imu
    # anonymous=True ensures that node name is unique
    rospy.init_node(SENSOR_NAME, anonymous=True)
    serial_port = rospy.get_param('~port', '/dev/ttyUSB0')
    serial_baud = rospy.get_param('~baudrate', 115200)
    # 10 Hz was small for imu's sampling rate, it was overshooting to the data
    # at the next sequence, so it's made 50 Hz
    rate = rospy.Rate(50)  # loop 50 times/second

    # Reading serial port with these parameters
    port = serial.Serial(serial_port, serial_baud, timeout=3.)

    # logdebug logs messages to rosout:
    rospy.logdebug("Using IMU sensor on port " + serial_port + " at baudrate"
                   + str(serial_baud))
    rospy.logdebug("Initializing sensor")

    imu_pub, magneto_pub = _get_topics(SENSOR_NAME)
    rospy.logdebug("Initialization complete")

    rospy.loginfo("Converting roll, pitch and yaw data to quaternions.")
    rospy.loginfo("Publishing roll, pitch, yaw data and orientations which is"
                  " quaternions.")

    imu_msg, magneto_msg = Imu(), MagneticField()
    imu_msg.header.frame_id, magneto_msg.header.frame_id = "imu", "magnetic_field"
    imu_msg.header.seq, magneto_msg.header.seq = 0, 0
    try:
        while not rospy.is_shutdown():
            line = port.readline().decode('utf-8')
            if line == '':
                rospy.logwarn("No data")
            else:
                if line.startswith('$VNYMR'):
                    # example line: $VNYMR,+092.530,-000.013,+000.277,-00.1505,-00.1516,+00.7189,-00.005,-00.0514
                    # order: Yaw, Pitch, Roll, Magnetic, Acceleration,
                    # and Angular Rate Measurements
                    data = line.split(',')

                    yaw = np.deg2rad(float(data[1]))
                    pitch = np.deg2rad(float(data[2]))
                    roll = np.deg2rad(float(data[3]))
                    mag_x, mag_y, mag_z = float(data[4]), float(data[5]), float(data[6])
                    acc_x, acc_y, acc_z = float(data[7]), float(data[8]), float(data[9])
                    gyro_x, gyro_y, gyro_z = float(data[10]), float(data[11]), float(data[12][:-7])

                    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

                    time = rospy.Time.now()
                    imu_msg.header.stamp = time # time stamp for the header
                    magneto_msg.header.stamp = time

                    imu_msg.linear_acceleration.x = acc_x
                    imu_msg.linear_acceleration.y = acc_y
                    imu_msg.linear_acceleration.z = acc_z

                    imu_msg.angular_velocity.x = gyro_x
                    imu_msg.angular_velocity.y = gyro_y
                    imu_msg.angular_velocity.z = gyro_z

                    imu_msg.orientation.x = quaternion[0]
                    imu_msg.orientation.y = quaternion[1]
                    imu_msg.orientation.z = quaternion[2]
                    imu_msg.orientation.w = quaternion[3]

                    magneto_msg.magnetic_field.x = mag_x
                    magneto_msg.magnetic_field.y = mag_y
                    magneto_msg.magnetic_field.z = mag_z

                    imu_msg.header.seq += 1
                    magneto_msg.header.seq += 1

                    imu_pub.publish(imu_msg)
                    magneto_pub.publish(magneto_msg)

                    print(50 * '-')
                    print("time: ", time)
                    print("yaw (rad) : ", yaw)
                    print("pitch (rad) : ", pitch)
                    print("roll (rad) : ", roll)
                    print("quaternion ", quaternion)
                    print("magnetic (Gauss) : ", mag_x, mag_y, mag_z)
                    print("acceleration (m/s^2) : ", acc_x, acc_y, acc_z)
                    print("gyro (rad/s): ", gyro_x, gyro_y, gyro_z)

            rate.sleep()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROSInterruptExecution. Shutting down GNSS node ...")
        port.close()

    except serial.serialutil.SerialException:
        rospy.loginfo("SerialException. Shutting down GNSS node ...")
    pass


if __name__ == '__main__':
    talker()

