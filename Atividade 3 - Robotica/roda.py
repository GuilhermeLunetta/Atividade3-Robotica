#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
from geometry_msgs.msg import Twist, Vector3
import numpy as np

if __name__ == "__main__":
    rospy.init_node("roda_exemplo")
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=3)

    k=0
    t=2.75

    vectorForward = Twist(Vector3(0.2, 0, 0), Vector3(0, 0, 0))
    vectorTurn = Twist(Vector3(0, 0, 0), Vector3(0, 0, np.pi/6))
    vectorStop = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))

    try:

        while k < 5:

            pub.publish(vectorForward)
            rospy.sleep(5)
            pub.publish(vectorStop)
            rospy.sleep(0.5)
            pub.publish(vectorTurn)
            rospy.sleep(t)
            pub.publish(vectorStop)
            k+=1
            t+=0.025

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")
