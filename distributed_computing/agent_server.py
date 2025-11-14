'''In this file you need to implement remote procedure call (RPC) server

* There are different RPC libraries for python, such as xmlrpclib, json-rpc. You are free to choose.
* The following functions have to be implemented and exported:
 * get_angle
 * set_angle
 * get_posture
 * execute_keyframes
 * get_transform
 * set_transform
* You can test RPC server with ipython before implementing agent_client.py
'''

# add PYTHONPATH
import os
import numpy as np
import sys
import threading
import xmlrpc.server as SimpleXMLRPCServer
import pickle
import time
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'kinematics'))

from inverse_kinematics import InverseKinematicsAgent

from xmlrpc.server import SimpleXMLRPCRequestHandler

class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

class ServerAgent(InverseKinematicsAgent):
    '''ServerAgent provides RPC service
    '''
    def __init__(self):
        super(ServerAgent, self).__init__()
        with open('joint_control\\robot_pose.pkl', 'rb') as f:
            self.posture_classifier = pickle.load(f)
        self.posture = 'unknown'
        
        self.server = SimpleXMLRPCServer.SimpleXMLRPCServer(('localhost', 8888), requestHandler=RequestHandler, allow_none=True)
        self.server.register_introspection_functions()
        self.server.register_multicall_functions()
        self.server.register_instance(self)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        print('RPC Server is ready')
        
    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        return super(ServerAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = 'unknown'
        data = []
        joints = ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch']
        for jn in joints:
            data.append(perception.joint[jn])
        data.append(perception.imu[0])
        data.append(perception.imu[1])
        pred = self.posture_classifier.predict(np.array(data).reshape(1, -1))
        posture = os.listdir('joint_control\\robot_pose_data_json')[pred[0]]

        return posture
    
    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        return self.perception.joint.get(joint_name)
    
    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller
        '''
        if joint_name in self.perception.joint:
            self.target_joints[joint_name] = angle
            print('Setting angle ', joint_name, 'to ', str(angle))
        else:
            print('Error: Failed to set angle')

    def get_posture(self):
        '''return current posture of robot'''
        return self.posture

    def execute_keyframes(self, keyframes):
        '''excute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        self.keyframes = keyframes

    def get_transform(self, name):
        '''get transform with given name
        '''
        return self.transforms[name]

    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        self.transforms[effector_name] = transform

if __name__ == '__main__':
    agent = ServerAgent()
    agent.run()

