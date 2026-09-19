"""Start all MVP nodes against one scene file."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration,PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scene=LaunchConfiguration("scene")
    return LaunchDescription([
        DeclareLaunchArgument("scene",default_value=PathJoinSubstitution([FindPackageShare("avm_bringup"),"config","scene.yaml"])),
        DeclareLaunchArgument("backend",default_value="gsplat"),
        DeclareLaunchArgument("render_hz",default_value="2.0"),
        DeclareLaunchArgument("port",default_value="8080"),
        DeclareLaunchArgument("host",default_value="127.0.0.1"),
        Node(package="scene_manager",executable="scene_manager_node",output="screen",parameters=[{"scene":scene}]),
        Node(package="gs_renderer",executable="gs_renderer_node",output="screen",parameters=[{
            "backend":LaunchConfiguration("backend"),"render_hz":LaunchConfiguration("render_hz")}]),
        Node(package="avm_stitcher",executable="avm_stitcher_node",output="screen"),
        Node(package="scene_editor",executable="scene_editor_node",output="screen",parameters=[{
            "port":LaunchConfiguration("port"),"host":LaunchConfiguration("host")}]),
    ])
