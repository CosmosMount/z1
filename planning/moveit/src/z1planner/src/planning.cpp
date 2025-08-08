#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>

class SimPosePlanner : public rclcpp::Node
{
public:
  SimPosePlanner(
      const moveit::planning_interface::MoveGroupInterfacePtr &mgi)
      : Node("sim_pose_planner"),
        move_group_interface_(mgi),
        last_pose_received_(false)
  {
    RCLCPP_INFO(this->get_logger(), "Initializing SimPosePlanner...");

    move_group_interface_->setPlanningTime(3.0);
    move_group_interface_->setNumPlanningAttempts(5);

    // 订阅目标位姿
    sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "sim_state", 10,
        std::bind(&SimPosePlanner::poseCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Waiting for sim_state messages...");
  }

private:
  void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (!last_pose_received_)
    {
      last_pose_ = *msg;
      last_pose_received_ = true;
      planAndExecute(msg);
      return;
    }

    // 计算位置差
    double dx = msg->pose.position.x - last_pose_.pose.position.x;
    double dy = msg->pose.position.y - last_pose_.pose.position.y;
    double dz = msg->pose.position.z - last_pose_.pose.position.z;
    double pos_dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    // 计算姿态差（四元数夹角）
    tf2::Quaternion q1, q2;
    tf2::fromMsg(last_pose_.pose.orientation, q1);
    tf2::fromMsg(msg->pose.orientation, q2);
    double angle_diff = q1.angleShortestPath(q2);

    constexpr double POS_THRESHOLD = 0.005;           // 5mm
    constexpr double ANGLE_THRESHOLD = 5.0 * M_PI / 180.0; // 5度

    if (pos_dist < POS_THRESHOLD && angle_diff < ANGLE_THRESHOLD)
    {
      RCLCPP_INFO(this->get_logger(),
                  "Pose change too small (pos: %.4f m, angle: %.2f deg), skipping planning.",
                  pos_dist, angle_diff * 180.0 / M_PI);
      return;
    }

    last_pose_ = *msg;
    planAndExecute(msg);
  }

  void planAndExecute(const geometry_msgs::msg::PoseStamped::SharedPtr &msg)
  {
    RCLCPP_INFO(this->get_logger(), "Planning and executing...");

    move_group_interface_->setPoseTarget(*msg);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = static_cast<bool>(move_group_interface_->plan(plan));

    if (success)
    {
      move_group_interface_->execute(plan);
      RCLCPP_INFO(this->get_logger(), "Execution complete.");
    }
    else
    {
      RCLCPP_ERROR(this->get_logger(), "Planning failed!");
    }
  }

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_;
  moveit::planning_interface::MoveGroupInterfacePtr move_group_interface_;

  geometry_msgs::msg::PoseStamped last_pose_;
  bool last_pose_received_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);

  // 创建 Node
  auto node = std::make_shared<rclcpp::Node>("sim_pose_planner_base");

  // 创建 MoveGroupInterface（这里 "Arm" 要改成你 SRDF 里定义的规划组名）
  auto mgi = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node, "Arm");

  // 创建实际 Planner 节点
  auto planner = std::make_shared<SimPosePlanner>(mgi);

  // 用 executor 同时 spin 两个节点（MGI 和 Planner）
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(planner);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
