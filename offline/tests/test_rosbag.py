from shmtrack.io.rosbag_reader import RosbagReader

bag = "/data/DEV/shm-vibration-tracking/data/WTT/e7_90rpm/e7_90rpm_run1.bag"
r = RosbagReader(bag, cams=["sony_cam1","sony_cam2","sony_cam3"])

for cam in ["sony_cam1","sony_cam2","sony_cam3"]:
    print(cam, r.topics_for_cam(cam), r.get_camera_info(cam) is not None)

fr = next(r.iter_frames("sony_cam1", max_frames=5))
print("first frame:", fr.cam, fr.index, fr.t_bag_sec, fr.t_hdr_sec, fr.image_bgr.shape)
