#!/usr/bin/env python3
"""
在 Gazebo 中采集 Dumpster 模型多视角图像。

使用方式：
1) 先执行 sros（已激活 ROS 环境）
2) 启动 gazebo（推荐使用 tool/model_capture/capture_dumpster.launch）
3) 运行：
   python3 tool/model_capture/capture_dumpster_dataset.py --output_dir ./images_e/dumpster --rings 4 --views_per_ring 24
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import cv2


class GazeboDumpsterCapture:
    def __init__(self, args):
        self.args = args

        import rospy
        from cv_bridge import CvBridge
        from gazebo_msgs.msg import ModelState
        from gazebo_msgs.srv import DeleteModel, SetModelState, SpawnModel
        from sensor_msgs.msg import Image
        from tf.transformations import quaternion_from_euler

        self.rospy = rospy
        self.ModelState = ModelState
        self.SpawnModel = SpawnModel
        self.DeleteModel = DeleteModel
        self.SetModelState = SetModelState
        self.Image = Image
        self.quaternion_from_euler = quaternion_from_euler

        self.bridge = CvBridge()
        self.latest_image = None
        self.image_stamp = None

        self.rospy.init_node("dumpster_capture", anonymous=True)
        self.image_sub = self.rospy.Subscriber(args.image_topic, self.Image, self._image_cb, queue_size=1)

        self._wait_services()
        self.spawn_model = self.rospy.ServiceProxy("/gazebo/spawn_sdf_model", self.SpawnModel)
        self.delete_model = self.rospy.ServiceProxy("/gazebo/delete_model", self.DeleteModel)
        self.set_model_state = self.rospy.ServiceProxy("/gazebo/set_model_state", self.SetModelState)

    def _wait_services(self):
        services = [
            "/gazebo/spawn_sdf_model",
            "/gazebo/delete_model",
            "/gazebo/set_model_state",
        ]
        for srv in services:
            self.rospy.loginfo(f"等待服务: {srv}")
            self.rospy.wait_for_service(srv, timeout=20)

    def _image_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.image_stamp = msg.header.stamp.to_sec()
        except Exception as exc:
            self.rospy.logwarn(f"图像转换失败: {exc}")

    def _delete_if_exists(self, model_name: str):
        try:
            self.delete_model(model_name)
            self.rospy.sleep(0.1)
        except Exception:
            pass

    def spawn_models(self):
        self._delete_if_exists(self.args.dumpster_name)
        self._delete_if_exists(self.args.camera_name)

        dumpster_sdf = Path(self.args.dumpster_sdf).expanduser()
        camera_sdf = Path(self.args.camera_sdf).expanduser()

        if not dumpster_sdf.exists():
            raise FileNotFoundError(f"找不到 dumpster sdf: {dumpster_sdf}")
        if not camera_sdf.exists():
            raise FileNotFoundError(f"找不到 camera sdf: {camera_sdf}")

        dumpster_xml = dumpster_sdf.read_text(encoding="utf-8")
        camera_xml = camera_sdf.read_text(encoding="utf-8")

        self.rospy.loginfo("生成 Dumpster 模型")
        self.spawn_model(self.args.dumpster_name, dumpster_xml, "", self._build_state(0, 0, 0, 0).pose, "world")

        self.rospy.loginfo("生成采集相机模型")
        self.spawn_model(self.args.camera_name, camera_xml, "", self._build_state(0, -self.args.radius_min, self.args.height_min, 0).pose, "world")

        self.rospy.sleep(1.0)

    def _build_state(self, x: float, y: float, z: float, yaw: float):
        state = self.ModelState()
        state.model_name = self.args.camera_name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        qx, qy, qz, qw = self.quaternion_from_euler(0.0, 0.0, yaw)
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
        return state

    @staticmethod
    def _look_at_yaw(camera_x: float, camera_y: float, target_x: float = 0.0, target_y: float = 0.0) -> float:
        return math.atan2(target_y - camera_y, target_x - camera_x)

    def _wait_for_image(self, min_wait=0.15, timeout=2.0):
        start = time.time()
        last_stamp = self.image_stamp
        while time.time() - start < timeout and not self.rospy.is_shutdown():
            if self.latest_image is not None and self.image_stamp != last_stamp:
                if time.time() - start >= min_wait:
                    return self.latest_image.copy()
            self.rospy.sleep(0.02)

        if self.latest_image is not None:
            return self.latest_image.copy()
        return None

    def capture(self):
        out_dir = Path(self.args.output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        random.seed(self.args.seed)
        index = self.args.start_index

        rings = max(1, self.args.rings)
        views_per_ring = max(3, self.args.views_per_ring)

        for ring in range(rings):
            if self.rospy.is_shutdown():
                break

            height = self.args.height_min
            if rings > 1:
                height = self.args.height_min + (self.args.height_max - self.args.height_min) * ring / (rings - 1)

            radius = self.args.radius_min
            if rings > 1:
                radius = self.args.radius_min + (self.args.radius_max - self.args.radius_min) * ring / (rings - 1)

            for i in range(views_per_ring):
                if self.rospy.is_shutdown():
                    break

                theta = 2.0 * math.pi * i / views_per_ring
                theta += random.uniform(-self.args.angle_jitter, self.args.angle_jitter)
                cur_radius = radius + random.uniform(-self.args.radius_jitter, self.args.radius_jitter)

                x = cur_radius * math.cos(theta)
                y = cur_radius * math.sin(theta)
                yaw = self._look_at_yaw(x, y)

                state = self._build_state(x, y, height, yaw)
                self.set_model_state(state)
                self.rospy.sleep(self.args.settle_time)

                frame = self._wait_for_image(min_wait=self.args.min_image_wait)
                if frame is None:
                    self.rospy.logwarn("未获取到图像，跳过该视角")
                    continue

                file_name = f"dumpster_{index:06d}.png"
                save_path = out_dir / file_name
                ok = cv2.imwrite(str(save_path), frame)
                if ok:
                    self.rospy.loginfo(f"保存: {save_path}")
                    index += 1
                else:
                    self.rospy.logwarn(f"保存失败: {save_path}")

        self.rospy.loginfo(f"采集完成，总计 {index - self.args.start_index} 张")


def build_parser():
    default_dumpster_sdf = str(Path.home() / ".gazebo/models/dumpster/model.sdf")
    default_camera_sdf = str(Path(__file__).parent / "camera_model/model.sdf")

    parser = argparse.ArgumentParser(description="Gazebo Dumpster 多视角采集工具")
    parser.add_argument("--output_dir", type=str, default="./images_e/dumpster", help="图片输出目录")
    parser.add_argument("--image_topic", type=str, default="/capture_camera/image_raw", help="相机图像话题")

    parser.add_argument("--dumpster_sdf", type=str, default=default_dumpster_sdf, help="Dumpster SDF 文件路径")
    parser.add_argument("--camera_sdf", type=str, default=default_camera_sdf, help="采集相机 SDF 文件路径")
    parser.add_argument("--dumpster_name", type=str, default="dumpster_target", help="Gazebo 中 Dumpster 模型名")
    parser.add_argument("--camera_name", type=str, default="capture_camera", help="Gazebo 中相机模型名")

    parser.add_argument("--rings", type=int, default=4, help="高度/半径层数")
    parser.add_argument("--views_per_ring", type=int, default=24, help="每层采集视角数")

    parser.add_argument("--radius_min", type=float, default=3.5, help="最小采集半径")
    parser.add_argument("--radius_max", type=float, default=5.0, help="最大采集半径")
    parser.add_argument("--height_min", type=float, default=0.7, help="最小高度")
    parser.add_argument("--height_max", type=float, default=1.8, help="最大高度")

    parser.add_argument("--angle_jitter", type=float, default=0.05, help="角度随机抖动（弧度）")
    parser.add_argument("--radius_jitter", type=float, default=0.1, help="半径随机抖动")
    parser.add_argument("--settle_time", type=float, default=0.25, help="移动后等待时间")
    parser.add_argument("--min_image_wait", type=float, default=0.12, help="最小图像等待时间")

    parser.add_argument("--start_index", type=int, default=1, help="文件编号起始")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        capturer = GazeboDumpsterCapture(args)
        capturer.spawn_models()
        capturer.capture()
    except Exception as exc:
        if "rospy" in str(exc):
            print("运行失败: 未检测到 ROS Python 环境。")
            print("请先执行: sros")
        else:
            print(f"运行失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
