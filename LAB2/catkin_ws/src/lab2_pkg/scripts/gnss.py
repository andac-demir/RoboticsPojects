#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import serial
from std_msgs.msg import Float64, String
import utm

def _get_lat_lon(lat, lat_dir, lon, lon_dir):
    '''
    Description:
    Given latitude and and longitude from the sensor measurements in mmhhss
    convert them to type float
    Parameters:
        lat (float): latitude measured by GNSS puck
        lat_dir (str)
        lon (float): longitude measured by GNSS puck
        lon_dir (str)
    Return:
        lat (float)
        lon (float)
    '''
    lat, lon = lat/100, lon/100
    lat_hour, lon_hour = int(lat), int(lon)
    lat_min, lon_min = 100*(lat-lat_hour), 100*(lon-lon_hour)
    lat = lat_hour + lat_min / 60
    lon = lon_hour + lon_min / 60
    if lat_dir == 'S':
        lat = -lat
    if lon_dir == 'W':
        lon = -lon
    return lat, lon

def _get_utm(lat, lon):
    '''
     Description:
     Given latitude and and longitude returns utm coordinates.
     Parameters:
         lat (float): latitude in UTM
         lat_dir (str)
         lon (float): longitude in UTM
         lon_dir (str)
     Return:
         easting (float)
         northing (float)
         zone_num (int)
         zone_let (str)
     '''
    easting, northing, zone_num, zone_let = utm.from_latlon(lat, lon)
    return easting, northing, zone_num, zone_let

def _get_topics(node):
    '''
    Description
    gnss node is publishing to topics using message type Float64
    queue_size limits amount of queued messages if any subscriber is not
    receiving them fast enough
    Parameters:
        node (str)
    Return:
        pubs (list of <rospy.Publisher>)
    '''
    lat_pub = rospy.Publisher(node + '/latitude', Float64, queue_size=3)
    lon_pub = rospy.Publisher(node + '/longitude', Float64, queue_size=3)
    alt_pub = rospy.Publisher(node + '/altitude', Float64, queue_size=3)
    east_pub = rospy.Publisher(node + '/utm_easting', Float64, queue_size=3)
    north_pub = rospy.Publisher(node + '/utm_northing', Float64, queue_size=3)
    zone_pub = rospy.Publisher(node + '/utm_zone', Float64, queue_size=3)
    let_pub = rospy.Publisher(node + '/utm_letter', String, queue_size=3)
    pubs = [lat_pub, lon_pub, alt_pub, east_pub, north_pub, zone_pub, let_pub]
    return pubs

def talker():
    '''
    Description
    Node publishes sensor measurements to different topics
    '''
    SENSOR_NAME = "gnss"
    # This tells rospy the name of your node: gnss
    # anonymous=True ensures that node name is unique
    rospy.init_node(SENSOR_NAME, anonymous=True)
    serial_port = rospy.get_param('~port', '/dev/ttyUSB1')
    serial_baud = rospy.get_param('~baudrate', 4800)
    rate = rospy.Rate(10) # loop 10 times/second

    # Reading serial port with these parameters
    port = serial.Serial(serial_port, serial_baud, timeout=3.)

    # logdebug logs messages to rosout:
    rospy.logdebug("Using GNSS sensor on port " + serial_port + " at baudrate"
                   + str(serial_baud))
    rospy.logdebug("Initializing sensor")

    pubs = _get_topics(SENSOR_NAME)
    rospy.logdebug("Initialization complete")

    rospy.loginfo("Converting wgs84 to utm coordinates.")
    rospy.loginfo("Publishing latitude, longitude, altitude, "
                  "utm_easting, utm_northing, utm_zone, "
                  "utm_letter")
    try:
        while not rospy.is_shutdown():
            line = str(port.readline())
            if line == '':
                rospy.logwarn("No data")
            else:
                if line.startswith("b'$GPGGA"):
                    # example line: $GPGGA,xxxxxxx.000,3235.8452,N,xxxxxxx.9157,W,1,08,1.0,210.5,M,-23.5,M,,0000*61
                    data = line.split(',')

                    lat, lon = float(data[2]), float(data[4])
                    lat_dir, lon_dir = data[3], data[5]
                    alt = float(data[8]) # altitude

                    lat, lon = _get_lat_lon(lat, lat_dir, lon, lon_dir)
                    easting, northing, zone_num, zone_let = _get_utm(lat, lon)

                    pubs[0].publish(lat)
                    pubs[1].publish(lon)
                    pubs[2].publish(alt)
                    pubs[3].publish(easting)
                    pubs[4].publish(northing)
                    pubs[5].publish(zone_num)
                    pubs[6].publish(zone_let)

                    print(50 * '-')
                    print("Latitude: ", lat)
                    print("Longitude: ", lon)
                    print("Altitude: ", alt)
                    print("Easting: ", easting)
                    print("Northing: ", northing)
                    print("Zone Number: ", zone_num)
                    print("Zone Letter: ", zone_let)

            rate.sleep()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROSInterruptExecution. Shutting down GNSS node ...")
        port.close()

    except serial.serialutil.SerialException:
        rospy.loginfo("SerialException. Shutting down GNSS node ...")


if __name__ == '__main__':
    talker()