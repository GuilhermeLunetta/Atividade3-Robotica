#! /usr/bin/env python
# -*- coding:utf-8 -*-


import rospy
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan

vectorStop = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
vectorForward = Twist(Vector3(0.2, 0, 0), Vector3(0, 0, 0))
vectorBackward = Twist(Vector3(-0.2, 0, 0), Vector3(0, 0, 0))
dist=0

def scaneou(dado):
	global dist
	distan = np.array(dado.ranges).round(decimals=2)
	dist = distan[0]
	#print("Faixa valida: ", dado.range_min , " - ", dado.range_max )
	#print("Leituras:")
	#print(np.array(dado.ranges).round(decimals=2))
	#print("Intensities")
	#print(np.array(dado.intensities).round(decimals=2))

if __name__=="__main__":

	rospy.init_node("le_scan")

	pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 3)
	recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)

	while not rospy.is_shutdown():
	
		if dist > 1.02:
			pub.publish(vectorForward)
			
		elif dist < 1:
			pub.publish(vectorBackward)
		