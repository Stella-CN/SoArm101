# SO-ARM101 Learning Workspace

基于 ROS 2 的 SO-ARM101 机械臂学习与开发工作空间。

该工程主要用于 SO-ARM101 的底层驱动、ROS 2 集成、运动学、控制、视觉以及后续 LeRobot 相关实验。

## Project Structure

```text
.
├── 3rdparty/
│   └── lerobot/          # LeRobot 第三方源码
├── src/
│   └── so101_core/       # SO-ARM101 核心功能包
├── build/                # ROS 2 / colcon 编译产物
├── install/              # ROS 2 工作空间安装目录
├── log/                  # colcon 编译日志
└── README.md