import cv2
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Empty
from geometry_msgs.msg import PoseStamped, Quaternion

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

from slam_toolbox.srv import SerializePoseGraph, SaveMap

class CameraCoverageMap(Node):
    def __init__(self):
        super().__init__('CameraCoverageMap')

        # Three maps we are keeping track of
        self.slam_occ = OccupancyGrid()
        self.camera_coverage_occ = OccupancyGrid()
        self.combined_occ = OccupancyGrid()

        # Subscribe to the SLAM map
        self.slam_map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.slam_map_callback,
            10
        )

        # Subscribe to empty message that indicates when camera scanning is done
        # TODO: presence of this indicates that this should probably be a ros2 service
        self.done_scanning_sub = self.create_subscription(
            Empty,
            "/done_scanning",
            self.update_coverage_callback,
            10
        )

        # Used to get current pose around which to draw a circle
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        tf_future = self.tf_buffer.wait_for_transform_async('map', 'base_link', rclpy.time.Time())
        rclpy.spin_until_future_complete(self, tf_future)
        
        # Publisher to view raw camera coverage map
        self.cam_map_raw_pub = self.create_publisher(OccupancyGrid,
                              "/camera_coverage_map_raw",
                              10)
        
        self.init_map_received = False


    def slam_map_callback(self, msg:OccupancyGrid) -> None:
        """
        Keep updating cam coverage info (size) and header as map info gets updated
        """
        self.get_logger().info("slam_callback")
        self.slam_occ = msg

        if not self.init_map_received:
            self.camera_coverage_occ.header = self.slam_occ.header
            self.camera_coverage_occ.info = self.slam_occ.info
            self.init_map_received = True

        og_width = self.camera_coverage_occ.info.width
        og_height = self.camera_coverage_occ.info.height

        if len(self.camera_coverage_occ.data) == 0:
            self.camera_coverage_occ.data = [-1] * \
                                        self.camera_coverage_occ.info.height * \
                                        self.camera_coverage_occ.info.width

        if og_width != self.slam_occ.info.width or \
            og_height != self.slam_occ.info.height:
            self.match_occupancy1_to_occupancy2(occupancy1=self.camera_coverage_occ,
                                                occupancy2=self.slam_occ)

    def update_coverage_callback(self, msg: Empty):
        """
        Add circles to the map every time camera scan is done
        """
        robot_pose_stamped = self.get_current_pose()
        
        print(self.camera_coverage_occ.header.frame_id)
        
        self.fill_circle_in_occupancy_grid(
            occupancy_grid=self.camera_coverage_occ,
            robot_x=robot_pose_stamped.pose.position.x,
            robot_y=robot_pose_stamped.pose.position.y,
            circle_radius_m=1.0,
            fill_color=100
        )

        self.cam_map_raw_pub.publish(self.camera_coverage_occ)

    def fill_circle_in_occupancy_grid(
        self,
        occupancy_grid,
        robot_x,
        robot_y,
        circle_radius_m,
        fill_color
    ):
        """
        Fills a circle of the specified radius (in meters) around (robot_x, robot_y)
        with the specified 'color' in the 'occupancy_grid.data' array.

        :param occupancy_grid: A ROS2 OccupancyGrid message.
        :param robot_x: The x-coordinate (in world/map frame) of the center of the circle.
        :param robot_y: The y-coordinate (in world/map frame) of the center of the circle.
        :param circle_radius_m: Radius of the circle in meters.
        :param color: The occupancy value to assign (e.g., 0..100 or -1).
        """

        # Extract grid parameters
        width       = occupancy_grid.info.width
        height      = occupancy_grid.info.height
        resolution  = occupancy_grid.info.resolution
        origin_x    = occupancy_grid.info.origin.position.x
        origin_y    = occupancy_grid.info.origin.position.y

        print(f"origin: ({origin_x}, {origin_y})")
        print("resolution", resolution)
        print(f"width, height: {width}, {height}", resolution)

        # input("waiting for input")
        # Convert center from world coords to grid indices
        center_x_idx = int((robot_x - origin_x) / resolution)
        center_y_idx = int((robot_y - origin_y) / resolution)

        # Convert circle radius from meters to grid cells
        radius_cells = int(circle_radius_m / resolution)

        # Precompute radius^2 for faster distance checks
        radius_sq = radius_cells * radius_cells

        # Define bounding box in grid coordinates
        min_x = max(0, center_x_idx - radius_cells)
        max_x = min(width - 1, center_x_idx + radius_cells)
        min_y = max(0, center_y_idx - radius_cells)
        max_y = min(height - 1, center_y_idx + radius_cells)

        # Modify occupancy_grid.data within the circle
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Calculate the squared distance from the center
                dist_sq = (x - center_x_idx)**2 + (y - center_y_idx)**2
                if dist_sq <= radius_sq:
                    # Convert (x,y) into the correct index for the 1D data array
                    index = y * width + x
                    try:
                        occupancy_grid.data[index] = fill_color 
                    except IndexError:
                        # print("oopsies")
                        continue
    
    def match_occupancy1_to_occupancy2(self, 
                                       occupancy1:OccupancyGrid, 
                                       occupancy2:OccupancyGrid):
        """
        Modifies occupancy1 *in place* so that its width, height, and origin 
        match occupancy2, but any cell in occupancy1 that had value == 100 
        remains at the same real-world (x,y) location in the new occupancy1.
        """

        # Extract relevant info from occupancy1 and occupancy2
        resolution1 = occupancy1.info.resolution
        resolution2 = occupancy2.info.resolution

        width1 = occupancy1.info.width
        height1 = occupancy1.info.height
        data1 = occupancy1.data

        # occupancy1's old origin
        origin_x1 = occupancy1.info.origin.position.x
        origin_y1 = occupancy1.info.origin.position.y

        # occupancy2's origin, width, height
        width2 = occupancy2.info.width
        height2 = occupancy2.info.height
        origin_x2 = occupancy2.info.origin.position.x
        origin_y2 = occupancy2.info.origin.position.y

        # For simplicity, assume the resolution is the same
        # if not math.isclose(resolution1, resolution2, rel_tol=1e-9):
        #     raise ValueError(
        #         "Grid resolutions differ. This code assumes the same resolution."
        #     )
        resolution = resolution1

        # Create a new data array for occupancy1, matching the size of occupancy2.
        # Use -1 (unknown) or 0 (free) as a default fill. (Pick whichever you prefer.)
        new_data = [-1] * (width2 * height2)

        # Remap old occupancy1 data into new_data based on real-world coords
        for row1 in range(height1):
            for col1 in range(width1):
                index1 = row1 * width1 + col1
                if data1[index1] == 100:
                    # Convert the old cell (col1, row1) in occupancy1 to real-world coords
                    world_x = origin_x1 + (col1 + 0.5) * resolution
                    world_y = origin_y1 + (row1 + 0.5) * resolution

                    # Convert real-world coords to new cell indices in the expanded occupancy1
                    col_new = int((world_x - origin_x2) // resolution)
                    row_new = int((world_y - origin_y2) // resolution)

                    # Check bounds in the new occupancy grid
                    if 0 <= col_new < width2 and 0 <= row_new < height2:
                        index_new = row_new * width2 + col_new
                        new_data[index_new] = 100

        # Now update occupancy1 *in place*
        occupancy1.data = new_data
        occupancy1.info.width = width2
        occupancy1.info.height = height2
        occupancy1.info.origin = occupancy2.info.origin
        # If needed, also ensure occupancy1.info.resolution = occupancy2.info.resolution
        occupancy1.info.resolution = resolution2


    def get_current_pose(self) -> PoseStamped:
        try:
            t = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time(), Duration(seconds=0.5))
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform odom to base_link: {ex}')
            self.get_logger().warn('Current pose unavailable.')
            return None
            
        p = PoseStamped()
        p.pose.position.x = t.transform.translation.x
        p.pose.position.y = t.transform.translation.y
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = 'odom'
        return p

def main(args=None):
    rclpy.init(args=args)
    node = CameraCoverageMap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()