# ros_socket_bridge.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import socket
import threading
import json

class SocketBridgeNode(Node):
    def __init__(self):
        super().__init__('socket_bridge')
        self.publisher_ = self.create_publisher(PoseStamped, 'sim_state', 10)
        self.subscription_ = self.create_subscription(String,'cmd_target',self.cmd_callback,10)
        self.client_conn = None
        threading.Thread(target=self.start_socket_server, daemon=True).start()

    def start_socket_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(('127.0.0.1', 9999))
        server.listen(1)
        self.get_logger().info("Waiting for IsaacSim connection...")
        conn, addr = server.accept()
        self.get_logger().info(f"Connected by {addr}")
        self.client_conn = conn

        buffer = ""
        while True:
            recv = conn.recv(1024).decode('utf-8')
            if not recv:
                self.get_logger().warn("Connection closed by client")
                break
            buffer += recv
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                msg = json.loads(line)
                
                if msg["type"] == "state":
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = "world"  # or "base_link" depending on your TF

                    # 你从 Web 或 Unity 发来的消息
                    data = msg["data"]

                    pose_msg.pose.position.x = data["position"][0]
                    pose_msg.pose.position.y = data["position"][1]
                    pose_msg.pose.position.z = data["position"][2]

                    pose_msg.pose.orientation.x = data["rotation"][0]
                    pose_msg.pose.orientation.y = data["rotation"][1]
                    pose_msg.pose.orientation.z = data["rotation"][2]
                    pose_msg.pose.orientation.w = data["rotation"][3]

                    self.publisher_.publish(pose_msg)

    def cmd_callback(self, msg):
        if self.client_conn:
            cmd = json.dumps({"type": "cmd", "data": json.loads(msg.data)}) + '\n'
            self.client_conn.sendall(cmd.encode('utf-8'))

def main(args=None):
    rclpy.init(args=args)
    node = SocketBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
