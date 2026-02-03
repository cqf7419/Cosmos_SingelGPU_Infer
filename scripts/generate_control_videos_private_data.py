import os
import pickle as pkl
import json
import numpy as np
from av_utils.render_config import CAMERA_NAME_MAPPING, SETTINGS
from loguru import logger
import open3d as o3d
import pickle as pkl
from PIL import Image
from pathlib import Path
import imageio

batchdir = "/mnt_gx/lidar_data/datae2e/union_cases_97430_default"
cyberlog = "GT3V3-00175_20250113104824_20250113104935_132728349"
meta_info_path = os.path.join(batchdir, "meta_infos", f"{cyberlog}.pkl")
dynamic_root_path = os.path.join(batchdir, "temp", cyberlog, "occ/preproc/dynamic/objects")
rtmap_path = os.path.join(batchdir, 'rtmap', cyberlog)
undistort_images_path = os.path.join(batchdir, "undistort_images", cyberlog)

category_map = {
    # "Car" / "Pedestrian" / "Cyclist" / "Truck"
    "V-Car": "Car",
    "V-LightTruck": "Car",#"Truck",
    "V-HeavyTruck": "Car",#"Truck",
    "V-Bus": "Car",#"Truck",
    "V-EngineeringVehicle ": "Car",#"Truck",
    "V-MicroCar": "Car",#"Truck",
    "V-XiaoG": "Car",#"Truck",
    "V-Other": "Car",

    "C-Motorcyclist": "Cyclist",
    "C-Motorcycle": "Cyclist",
    "C-Bicyclist": "Cyclist",
    "C-Bicycle": "Cyclist",
    "C-Tricycle": "Cyclist",
    "C-ChildVehicle": "Cyclist",
    "C-Other": "Cyclist",

    "H-Pedestrian": "Pedestrian",
    "H-VehPedestrian": "Pedestrian",
    "H-NonStandingPedestrian": "Pedestrian",
    "H-Child": "Pedestrian",
    "H-Other": "Pedestrian",
}

def filter_pcd(pcd, r = 0.35, d = 3):
    from scipy.spatial import KDTree
    radius = r
    tree = KDTree(pcd)
    density = np.array([len(tree.query_ball_point(point,radius)) for point in pcd])
    condition = density>=d
    marked = np.zeros_like(density,dtype=bool)
    marked[condition] = True
    return marked

def load_obstacle_data():
    '''
    加载“机非人”实例信息
    '''
    frame_obstacles = {}
    baselidar2world = {}
    frame_id_to_timestamp = {}
    frame_data = pkl.load(open(meta_info_path, "rb"))
    for frame_ind in range(len(frame_data['frames'])):
        time_stamp = frame_data['frames'][frame_ind]['log_time_stamp']
        frame_id_to_timestamp[frame_ind] = str(time_stamp)
        frame_obstacles[str(time_stamp)] = {}
        baselidar2world[str(time_stamp)] = frame_data['frames'][frame_ind]['optimized_pose']

    obstacle_id_list = os.listdir(dynamic_root_path) 
    for obstacle_id in obstacle_id_list:
        dynamic_obj_path = os.path.join(dynamic_root_path, obstacle_id)
        info_json = os.path.join(dynamic_obj_path, "info.json")

        with open(info_json, 'r') as file:
            datas = json.load(file)
            data = datas["tracks"]
            if datas["sem_name"] in category_map:
                object_type = category_map[datas["sem_name"]]
            else:
                raise ValueError(f"{datas['sem_name']} not in category_map")

        pcd_path = os.path.join(dynamic_obj_path, 'vertices.txt')
        if os.path.exists(pcd_path):
            obj_pcd = np.loadtxt(pcd_path).astype(np.float32)
        else:
            pcd_path = os.path.join(dynamic_obj_path, 'stitch.pcd')
            obj_pcd = np.array(o3d.io.read_point_cloud(pcd_path).points, dtype=np.float32)
        dense_mask = filter_pcd(obj_pcd, r = 0.35, d = 5) # 过滤离群点 使bbox更紧凑
        obj_point = obj_pcd[dense_mask]
        min_bbox = np.min(obj_point, axis=0)  
        max_bbox = np.max(obj_point, axis=0)
        length = max_bbox[0] - min_bbox[0]  # x 方向的差值
        width = max_bbox[1] - min_bbox[1]   # y 方向的差值
        height = max_bbox[2] - min_bbox[2]  # z 方向的差值

        for key, value in data.items():
            b2l = np.array(value["T_b2l"], dtype=np.float32).reshape(4,4)
            object_to_world = baselidar2world[key] @ b2l
            frame_obstacles[key][str(obstacle_id)] = {
                    "object_to_world": object_to_world.tolist(),
                    "object_lwh": [length, width, height],
                    "object_type": object_type,
                    "object_is_moving": True,
                }
    return frame_obstacles, frame_id_to_timestamp

from av_utils.camera.ftheta import FThetaCamera
from av_utils.camera.pinhole import PinholeCamera
import cv2
def load_camera_from_calibration_estimate(
    camera_names: list[str],
    resize_hw: tuple[int, int],
):
    '''
    input:
        camera_names 存储相机名的列表
        resize_hw 期望输出的长宽
    output:
        camera_dicts 字典 key为相机名 value为相机模型类 PinholeCamera或FThetaCamera
    '''
    all_camera_poses = dict()
    all_camera_models = dict()
    meta_info = pkl.load(open(meta_info_path, "rb"))
    for camera_channel in camera_names: 
        if 'PANO' in camera_channel:
            intrinsic = meta_info['frames'][0]['cams_info']['CAMERA_FRONT']["camera_intrinsics"][:3, :3] # TODO 暂时不支持鱼眼
        else:
            intrinsic = meta_info['frames'][0]['cams_info'][camera_channel]["camera_intrinsics"][:3, :3]
        img_path = os.path.join(batchdir, meta_info['frames'][0]['cams_info'][camera_channel]["data_path"])
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img_data = Image.fromarray(original_img)
        width, height = img_data.size
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        camera_model = PinholeCamera(fx, fy, cx, cy, width, height)

        resize_w, resize_h = resize_hw
        rescale_h = resize_h / camera_model.height
        rescale_w = resize_w / camera_model.width
        camera_model.rescale(rescale_h, rescale_w)
        # camera_models[camera_channel] = camera_model

        camera_poses = []
        for frame_data in meta_info['frames']:
            lidar2world = frame_data['optimized_pose']
            cam2lidar = frame_data["cams_info"][camera_channel]["compensated_camera2lidar"]
            cam2world = lidar2world @ cam2lidar
            camera_poses.append(cam2world)

        all_camera_poses[camera_channel] = np.array(camera_poses)
        all_camera_models[camera_channel] = camera_model

    return all_camera_poses, all_camera_models
    

from av_utils.clip_gt_io import interpolate_polyline
from av_utils.graphics_utils import (
    BoundingBox2D,
    LineSegment2D,
    Polygon2D,
    TriangleList2D,
    render_geometries,
)
from av_utils.laneline_utils import prepare_laneline_geometry_data
from av_utils.minimap_utils import (
    cuboid3d_to_polyline,
    get_type_from_name,
    load_hdmap_colors,
)
from av_utils.pcd_utils import (
    filter_by_height_relative_to_ego,
    interpolate_polyline_to_points,
    triangulate_polygon_3d,
)
from av_utils.bbox_utils import (
    build_cuboid_bounding_box,
    load_bbox_colors,
    simplify_type_in_object_info,
)
def load_map_data():
    '''
    output:
        all_map_data 列表, 里面每个元素代表每帧的rtmap_data的字典,
    '''
    all_map_data = []
    for json_file in os.listdir(rtmap_path):
        if "road_info" in json_file:
            continue
        json_dir = os.path.join(rtmap_path, json_file)
        with open(json_dir, 'r', encoding='utf-8') as f:
            rtmap_data = json.load(f)
        map_data = {
            "lanes": [],  # For Cosmos compatibility
            "lanelines": [],  # List of tuples: (polyline, type_string)
            "road_boundaries": [],
            "crosswalks": [],
            "traffic_lights": [],
            "traffic_signs": [],
            "poles": [],
            "road_markings": [],
            "wait_lines": [],
            "intersection_areas": [],
            "road_islands": [],
        }
        if 'lidar_to_body' in rtmap_data and 'body_to_map' in rtmap_data:
            lidar2body = np.array(rtmap_data['lidar_to_body']['data']).reshape(4,4)
            body2map = np.array(rtmap_data['body_to_map']['data']).reshape(4,4)
        else:
            raise ValueError(f'key error of {json_dir}')
        lidar2map = body2map@lidar2body
        if 'divide_line' in rtmap_data:
            divide_lines = rtmap_data['divide_line']
            for divide_line in divide_lines:
                is_virtual = divide_line["is_virtual"]
                if is_virtual: continue 
                origin_type = divide_line["origin_type"]
                points_in_line = divide_line["point"]
                points = []
                for xyz in points_in_line:
                    x = xyz['x']
                    y = xyz['y']
                    z = xyz['z']
                    points.append([x,y,z])
                points_lidar = np.array(points)
                point_world = (np.pad(points_lidar[...,:3], ((0,0),(0, 1)), constant_values=1) @ lidar2map.T)[:,:3]
                if origin_type == 1:
                    map_data["lanelines"].append((point_world, "WHITE SOLID_SINGLE"))
                elif origin_type == 2:
                    map_data["lanelines"].append((point_world, "WHITE LONG_DASHED_SINGLE"))
                elif origin_type == 3:
                    map_data["lanelines"].append((point_world, "WHITE SOLID_GROUP"))
                else:
                    map_data["lanelines"].append((point_world, "WHITE OTHER"))
        
        # if 'junction' in rtmap_data: # 路口
        #     junctions = rtmap_data['junction']
        #     for junction in junctions:
        #         points_in_junction = junction["point"]
        #         for xyz in points_in_junction:
        #             x = xyz['x']
        #             y = xyz['y']
        #             z = xyz['z']

        if 'stop_line' in rtmap_data:
            stop_lines = rtmap_data['stop_line']
            for stop_line in stop_lines:
                points_in_stop_line = stop_line["point"]
                points = []
                for xyz in points_in_stop_line:
                    x = xyz['x']
                    y = xyz['y']
                    z = xyz['z']
                    points.append([x,y,z])
                points_lidar = np.array(points)
                point_world = (np.pad(points_lidar[...,:3], ((0,0),(0, 1)), constant_values=1) @ lidar2map.T)[:,:3]
                map_data["wait_lines"].append(point_world)
        
        if 'cross_walk' in rtmap_data:
            cross_walks = rtmap_data['cross_walk']
            for cross_walk in cross_walks:
                points_in_cross_walk = cross_walk["point"]
                points = []
                for xyz in points_in_cross_walk:
                    x = xyz['x']
                    y = xyz['y']
                    z = xyz['z']
                    points.append([x,y,z])
                points_lidar = np.array(points)
                point_world = (np.pad(points_lidar[...,:3], ((0,0),(0, 1)), constant_values=1) @ lidar2map.T)[:,:3]
                map_data["crosswalks"].append(point_world)

        if 'object_line' in rtmap_data:
            object_lines = rtmap_data['object_line']
            for object_line in object_lines:
                points_in_object_line = object_line["point"]
                points = []
                for xyz in points_in_object_line:
                    x = xyz['x']
                    y = xyz['y']
                    z = xyz['z']
                    points.append([x,y,z])
                points_lidar = np.array(points)
                point_world = (np.pad(points_lidar[...,:3], ((0,0),(0, 1)), constant_values=1) @ lidar2map.T)[:,:3]
                map_data["road_boundaries"].append(point_world)
        
        all_map_data.append(map_data)
    return all_map_data

def create_minimap_geometry_objects_from_data(
    map_data: dict,
    camera_pose: np.ndarray,
    camera_model: PinholeCamera,
    hdmap_color_version: str = "v3",
    camera_pose_init: np.ndarray | None = None,
) -> list:
    """
    针对GT数据做过一次适配
    Build geometry objects for minimap layers for a single frame.

    Args:
        map_data: dict[name -> list[np.ndarray]], map data
        camera_pose: np.ndarray (4,4), camera pose
        camera_model: PinholeCamera, camera model
        hdmap_color_version: str, HD map color version
        camera_pose_init: Optional[np.ndarray], initial camera pose for height filtering

    Returns:
        list: geometry objects (LineSegment2D/Polygon2D/TriangleList2D)
    """
    minimap_to_rgb = load_hdmap_colors(hdmap_color_version)
    all_geometry_objects = []

    for minimap_name, elements in map_data.items():
        if not elements:
            continue

        # Skip 'lanes' if not in color config (it's often just for compatibility)
        if minimap_name == "lanes" and minimap_name not in minimap_to_rgb:
            continue

        # Special handling for lanelines with geometric patterns in V3
        if minimap_name == "lanelines" and elements and isinstance(elements[0], tuple):
            # 将原始车道线数据（带类型标签的折线）预处理为适合可视化渲染（特别是 V3 渲染器）的结构化格式，包括颜色、线宽、虚实线模式（pattern）等信息
            processed_lanelines = prepare_laneline_geometry_data(elements)

            # Render each laneline with its pattern
            for laneline_info in processed_lanelines:
                # 应用的是实时地图 不需要做过滤筛选 一定是当前可视范围的
                # Each laneline can have multiple segment lists (e.g., dual lines)
                for segments in laneline_info["pattern_segments_list"]:
                    if len(segments) == 0:
                        continue

                    # Project segments to camera space
                    xy_and_depth = camera_model.get_xy_and_depth(segments.reshape(-1, 3), camera_pose)
                    # Convert tensor to numpy if needed
                    if not isinstance(xy_and_depth, np.ndarray):
                        xy_and_depth = xy_and_depth.numpy()
                    xy_and_depth = xy_and_depth.reshape(-1, 2, 3)

                    # Filter valid line segments (both vertices in front of camera)
                    valid_line_segment_vertices = xy_and_depth[:, :, 2] >= 0
                    valid_line_segment_indices = np.all(valid_line_segment_vertices, axis=1)
                    valid_xy_and_depth = xy_and_depth[valid_line_segment_indices]

                    if len(valid_xy_and_depth) > 0:
                        all_geometry_objects.append(
                            LineSegment2D(
                                valid_xy_and_depth,
                                base_color=laneline_info["rgb_float"],
                                line_width=laneline_info["line_width"],
                            )
                        )
            continue  # Skip the normal processing for lanelines

        # Extract polylines from tuples if needed (for backward compatibility)
        if minimap_name == "lanelines" and elements and isinstance(elements[0], tuple):
            polylines = [polyline for polyline, _ in elements]
        else:
            polylines = elements

        # Skip if element type not defined
        try:
            minimap_type = get_type_from_name(minimap_name)
        except ValueError:
            # Skip unknown minimap types
            continue

        if minimap_type == "polyline":
            line_segment_list = []
            for polyline in polylines:
                # 应用的是实时地图 不需要做过滤筛选 一定是当前可视范围的
                # Subdivide the polyline for smooth rendering
                if minimap_name in ["lanelines", "road_boundaries"]:
                    polyline_subdivided = interpolate_polyline_to_points(polyline, segment_interval=0.8)
                else:
                    polyline_subdivided = polyline

                if len(polyline_subdivided) < 2:
                    continue

                # Create line segments
                line_segment = np.stack([polyline_subdivided[:-1], polyline_subdivided[1:]], axis=1)
                line_segment_list.append(line_segment)

            if len(line_segment_list) == 0:
                continue

            all_line_segments = np.concatenate(line_segment_list, axis=0)
            xy_and_depth = camera_model.get_xy_and_depth(all_line_segments.reshape(-1, 3), camera_pose)
            # Convert tensor to numpy if needed
            if not isinstance(xy_and_depth, np.ndarray):
                xy_and_depth = xy_and_depth.numpy()
            xy_and_depth = xy_and_depth.reshape(-1, 2, 3)

            # Filter valid line segments
            valid_line_segment_vertices = xy_and_depth[:, :, 2] >= 0
            valid_line_segment_indices = np.all(valid_line_segment_vertices, axis=1)
            valid_xy_and_depth = xy_and_depth[valid_line_segment_indices]

            if len(valid_xy_and_depth) > 0:
                color_float = np.array(minimap_to_rgb[minimap_name]) / 255.0
                all_geometry_objects.append(
                    LineSegment2D(
                        valid_xy_and_depth,
                        base_color=color_float,
                        line_width=5 if minimap_name == "poles" else 12,
                    )
                )

        elif minimap_type == "polygon" or minimap_type == "cuboid3d":
            for polygon in polylines:
                if minimap_type == "cuboid3d":
                    # Convert cuboid to polyline for rendering
                    polygon_converted = cuboid3d_to_polyline(polygon)
                else:
                    polygon_converted = polygon

                if minimap_name == "crosswalks":
                    # Subdivide and triangulate crosswalks
                    polygon_subdivided = interpolate_polyline_to_points(polygon_converted, segment_interval=0.8)
                    triangles_3d = triangulate_polygon_3d(polygon_subdivided)

                    if len(triangles_3d) == 0 or triangles_3d.size == 0:
                        continue

                    triangles_proj = camera_model.get_xy_and_depth(triangles_3d.reshape(-1, 3), camera_pose)
                    # Convert tensor to numpy if needed
                    if not isinstance(triangles_proj, np.ndarray):
                        triangles_proj = triangles_proj.numpy()
                    triangles_proj = triangles_proj.reshape(-1, 3, 3)
                    # Filter out triangles behind camera
                    invalid_triangles_indices = np.all(triangles_proj[:, :, 2] < 0, axis=1)
                    valid_triangles_indices = ~invalid_triangles_indices

                    if valid_triangles_indices.sum() > 0:
                        color_float = np.array(minimap_to_rgb[minimap_name]) / 255.0
                        all_geometry_objects.append(
                            TriangleList2D(
                                triangles_proj[valid_triangles_indices],
                                base_color=color_float,
                            )
                        )
                else:
                    # Regular polygon rendering
                    polygon_xy_and_depth = camera_model.get_xy_and_depth(polygon_converted, camera_pose)
                    # Convert tensor to numpy if needed
                    if not isinstance(polygon_xy_and_depth, np.ndarray):
                        polygon_xy_and_depth = polygon_xy_and_depth.numpy()

                    if not np.all(polygon_xy_and_depth[:, 2] < 0):
                        color_float = np.array(minimap_to_rgb[minimap_name]) / 255.0
                        all_geometry_objects.append(
                            Polygon2D(
                                polygon_xy_and_depth,
                                base_color=color_float,
                            )
                        )

    return all_geometry_objects

def create_bbox_geometry_objects_for_frame(
    current_object_info: dict,
    camera_pose: np.ndarray,
    camera_model: PinholeCamera,
    bbox_color_version: str = "v3",
    fill_face: str = "all",
    fill_face_style: str = "solid",
    line_width: int = 4,
    edge_color: list | None = None,
) -> list:
    """
    Build BoundingBox2D geometry objects for a single frame.

    Args:
        current_object_info: dict, object info for current frame
        camera_pose: np.ndarray (4,4), camera pose
        camera_model: PinholeCamera, camera model
        bbox_color_version: str, bbox color version
        fill_face: str, which faces to fill
        fill_face_style: str, style of face filling
        line_width: int, line width for rendering
        edge_color: list, optional edge color

    Returns:
        list[BoundingBox2D]: geometry objects for the current frame
    """
    # Build per-vertex color map
    gradient_class_colors = load_bbox_colors(bbox_color_version)
    object_type_to_per_vertex_color = {}

    for object_type, colors in gradient_class_colors.items():
        if isinstance(colors, list) and len(colors) == 2 and isinstance(colors[0], list):
            # Gradient color
            per_vertex_color = np.zeros((8, 3))
            per_vertex_color[[0, 1, 4, 5]] = np.array(colors[0]) / 255.0
            per_vertex_color[[2, 3, 6, 7]] = np.array(colors[1]) / 255.0
        else:
            # Uniform color
            per_vertex_color = np.tile(np.array(colors) / 255.0, (8, 1))
        object_type_to_per_vertex_color[object_type] = per_vertex_color

    edge_color_array = None
    if edge_color is not None:
        edge_color_array = np.array(edge_color) / 255.0

    # Store the 8 corner vertices of each object type
    object_type_to_corner_vertices = {"Car": [], "Truck": [], "Pedestrian": [], "Cyclist": [], "Others": []}

    tracking_ids = list(current_object_info.keys())
    tracking_ids.sort()

    for tracking_id in tracking_ids:
        object_info = current_object_info[tracking_id]
        object_info = simplify_type_in_object_info(object_info)

        object_to_world = np.array(object_info["object_to_world"])
        object_lwh = np.array(object_info["object_lwh"])
        cuboid_eight_vertices = build_cuboid_bounding_box(object_lwh[0], object_lwh[1], object_lwh[2], object_to_world)

        # Cull objects entirely behind camera
        if np.all(np.dot(cuboid_eight_vertices - camera_pose[:3, 3], camera_pose[:3, 2]) < 0):
            continue

        if object_info["object_type"] in ["Car", "Truck", "Pedestrian", "Cyclist"]:
            object_type_to_corner_vertices[object_info["object_type"]].append(cuboid_eight_vertices)
        else:
            object_type_to_corner_vertices["Others"].append(cuboid_eight_vertices)

    # Draw the bbox projection
    geometry_objects = []
    for object_type, all_corner_vertices in object_type_to_corner_vertices.items():
        if len(all_corner_vertices) == 0:
            continue

        n_objects = len(all_corner_vertices)
        all_corner_vertices_flatten = np.array(all_corner_vertices).reshape(-1, 3)
        all_points_in_cam = camera_model.transform_points(all_corner_vertices_flatten, np.linalg.inv(camera_pose))
        all_depth = all_points_in_cam[:, 2:3]
        all_xy = camera_model.ray2pixel(all_points_in_cam)
        all_xy_and_depth = np.hstack([all_xy, all_depth]).reshape(n_objects, 8, 3)

        # Valid corner: (1) 0 <= x <= width, (2) 0 <= y <= height, (3) depth > 0
        valid_x_mask = (all_xy_and_depth[:, :, 0] >= 0) & (all_xy_and_depth[:, :, 0] < camera_model.width)
        valid_y_mask = (all_xy_and_depth[:, :, 1] >= 0) & (all_xy_and_depth[:, :, 1] < camera_model.height)
        valid_depth_mask = all_xy_and_depth[:, :, 2] > 0
        not_valid_vertex_mask = ~valid_x_mask | ~valid_y_mask | ~valid_depth_mask
        not_valid_object_mask = np.all(not_valid_vertex_mask, axis=1)
        valid_object_mask = ~not_valid_object_mask

        all_xy_and_depth = all_xy_and_depth[valid_object_mask]

        for xy_and_depth in all_xy_and_depth:
            geometry_objects.append(
                BoundingBox2D(
                    xy_and_depth=xy_and_depth,
                    base_color_or_per_vertex_color=object_type_to_per_vertex_color[object_type],
                    fill_face=fill_face,
                    fill_face_style=fill_face_style,
                    line_width=line_width,
                    edge_color=edge_color_array,
                )
            )

    return geometry_objects

def render_hdmap_v3(
    camera_poses: np.ndarray,
    all_object_info: dict,
    map_data: dict,
    camera_model: PinholeCamera,
    frame_id_to_timestamp: dict,
    render_frame_ids: list[int] | None = None,
    hdmap_color_version: str = "v3",
    bbox_color_version: str = "v3",
    traffic_light_color_version: str = "v2",
    enable_height_filter: bool = False,
    device_index: int = 0,
) -> np.ndarray:
    """
    Render HD map using V3 moderngl-based rendering with proper depth occlusion.

    Args:
        camera_poses: np.ndarray, shape (N, 4, 4), camera poses
        all_object_info: dict, containing all object info
        map_data: dict, map data
        camera_model: PinholeCamera, camera model
        render_frame_ids: list[int], frame ids to render
        hdmap_color_version: str, HD map color version
        bbox_color_version: str, bbox color version
        traffic_light_color_version: str, traffic light color version
        enable_height_filter: bool, whether to filter elements by height
        device_index: int, device index

    Returns:
        np.ndarray, shape (N, H, W, 3), rendered frames
    """
    if render_frame_ids is None:
        render_frame_ids = list(range(len(camera_poses)))

    combined_frames = []

    # Get initial camera pose for height filtering
    camera_pose_init = camera_poses[render_frame_ids[0]] if enable_height_filter else None

    # # Prepare traffic light status data if enabled
    # tl_position_list = None
    # tl_status_dict = None
    # tl_status_to_rgb = None
    # if map_data.get("traffic_lights"):
    #     tl_position_list, tl_status_dict, tl_status_to_rgb = prepare_traffic_light_status_data_clipgt(
    #         map_data["traffic_lights"],
    #         traffic_light_color_version=traffic_light_color_version,
    #     )
    #     logger.info(f"Traffic light rendering enabled with {traffic_light_color_version} colors")

    logger.info(
        f"Rendering {len(render_frame_ids)} frames with V3 renderer "
        f"(hdmap: {hdmap_color_version}, bbox: {bbox_color_version})"
    )

    for frame_id in render_frame_ids:
        camera_pose = camera_poses[frame_id]

        # Build all geometry objects for this frame
        geometry_objects = []

        # Add minimap layers (excluding traffic lights if status rendering is enabled)
        # map_data_filtered = map_data.copy()
        # if tl_position_list is not None:
        #     # Remove traffic lights from regular minimap rendering
        #     map_data_filtered = {k: v for k, v in map_data.items() if k != "traffic_lights"}

        geometry_objects.extend(
            create_minimap_geometry_objects_from_data(
                map_data[frame_id], # TODO 验证一下可行性
                camera_pose,
                camera_model,
                hdmap_color_version,
                camera_pose_init=camera_pose_init,
            )
        )

        # Add bounding boxes  # TODO here
        current_object_info = all_object_info[frame_id_to_timestamp[frame_id]]
        geometry_objects.extend(
            create_bbox_geometry_objects_for_frame(
                current_object_info,
                camera_pose,
                camera_model,
                bbox_color_version,
                fill_face="all",
                fill_face_style="solid",
                line_width=4,
                edge_color=[200, 200, 200],
            )
        )

        # Render all geometries with proper depth ordering
        combined_frame = render_geometries(
            geometry_objects,
            camera_model.height,
            camera_model.width,
            depth_max=200,
            depth_gradient=True,
            device_index=device_index,
        )
        combined_frames.append(combined_frame)

    return np.stack(combined_frames, axis=0)

def save_control_video(
    rendered_frames: np.ndarray, save_root: Path, camera_name: str, fps: int = 30
) -> Path:
    """Save rendered frames as MP4 video with proper naming convention."""
    # Create clip directory
    save_root.mkdir(parents=True, exist_ok=True)
    output_file = save_root / f"{camera_name}.mp4"

    # Ensure video dimensions are even (required for x264)
    height, width = rendered_frames.shape[1:3]
    if height % 2 != 0:
        rendered_frames = rendered_frames[:, :-1, :]
    if width % 2 != 0:
        rendered_frames = rendered_frames[:, :, :-1]

    # Save video
    writer = imageio.get_writer(
        str(output_file),
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", "18", "-preset", "slow"],
    )

    for frame in rendered_frames:
        writer.append_data(frame)
    writer.close()

    return output_file

if __name__ == "__main__":
    import pdb
    # camera_names_list = ['CAMERA_FRONT']
    # camera_names_list = ['CAMERA_LEFT_BACK', 'CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_BACK', 'CAMERA_RIGHT_FRONT'] 
    camera_names_list = ['CAMERA_PANO_BACK', 'CAMERA_PANO_FRONT']

    # logger.info("加载障碍物“机非人”...")
    # all_object_info, frame_id_to_timestamp = load_obstacle_data()
    # # pdb.set_trace()

    # logger.info("加载相机内外参...")
    # camera_poses, camera_models  =load_camera_from_calibration_estimate(camera_names=camera_names_list, resize_hw=SETTINGS["RESIZE_RESOLUTION"])
    # # pdb.set_trace()

    # logger.info("加载实时地图信息...")
    # map_data = load_map_data()
    # # # pdb.set_trace()

    # for i, camera_name in enumerate(camera_names_list):
    #     logger.info(f"渲染相机 {i + 1}/{len(camera_names_list)}: {camera_name}")
    #     rendered_frames = render_hdmap_v3(
    #             camera_poses[camera_name],
    #             all_object_info,
    #             map_data,
    #             camera_models[camera_name],
    #             frame_id_to_timestamp
    #     )

    #     # Save video with new format
    #     save_root = Path("/data1/cqf/cosmos-transfer2.5-main/outputs") / batchdir.split('/')[-1] / cyberlog
    #     output_file = save_control_video(
    #         rendered_frames, save_root, camera_name, fps=SETTINGS["TARGET_RENDER_FPS"]
    #     )
    #     logger.info(f"Saved: {output_file}")

    for i, camera_name in enumerate(camera_names_list):
        save_root = Path("/data1/cqf/cosmos-transfer2.5-main/outputs") / batchdir.split('/')[-1] / cyberlog
        origin_video = []
        for img_path in os.listdir(undistort_images_path):
            img_path = os.path.join(undistort_images_path, img_path, f"{camera_name}.jpg")
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            original_img = cv2.resize(original_img, (1280, 720), interpolation=cv2.INTER_LINEAR) 
            zero_numpy = np.zeros_like(original_img)
            if "PANO" in camera_name:
                origin_video.append(zero_numpy)
            else:
                origin_video.append(original_img)
        writer = imageio.get_writer(
            save_root/f"input_{camera_name}.mp4",
            fps=SETTINGS["TARGET_RENDER_FPS"],
            codec="libx264",
            macro_block_size=None,
            ffmpeg_params=["-crf", "18", "-preset", "slow"],
        )   
        for img in origin_video:
            writer.append_data(img)
        writer.close()

