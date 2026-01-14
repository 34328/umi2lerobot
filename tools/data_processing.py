import numpy as np
import h5py
import scipy.interpolate as si
import scipy.spatial.transform as st



def auto_match(array1, array2, array1_ts, array2_ts):

    def get_closest_indices(ref_timestamps, query_timestamps):
    # For each element in query_timestamps, find which index in ref_timestamps is closest
        return np.array([np.abs(ref_timestamps - t).argmin() for t in query_timestamps])

    if array1.shape[0] < array2.shape[0]:
        idxs = get_closest_indices(array2_ts, array1_ts)
        # Now pick out the rows from head_data that best match each zed timestamp
        array2 = array2[idxs]
    
    else:
        idxs = get_closest_indices(array1_ts, array2_ts)
        
        array1 = array1[idxs]
        
    # check that the lengths are the same
    assert array1.shape[0] == array2.shape[0], f"Timestamps do not match after auto-matching. Shapes: {array1.shape}, {array2.shape}"
    
    # print(f"Auto-matched timestamps. New shapes: {array1.shape}, {array2.shape}")
    
    return array1, array2



def align_all_data(ep):
    """
    Align all data fields in a DexWild episode to a common frame count.
    
    Args:
        ep: HDF5 episode group
        
    Returns:
        dict: Dictionary with aligned data for each field
    """
    # 单手任务数据集的字段配置
    # 有新任务时修改这里
    numeric_fields = [
        ('right_leapv2', 'right_leapv2'),
        ('right_manus', 'right_manus'),
        ('right_tracker', 'head_right_tracker'),
        ('right_tracker', 'right_tracker_world'),
        ('right_tracker', 'zed_right_tracker'),
        ('zed', 'zed_pose'),
        ('zed', 'zed_ts'),
    ]
    
    # 相机列表
    image_fields = ['head_cam', 'right_pinky_cam', 'right_thumb_cam', 'zed_obs']
    
    # Step 1: Load all numeric data and find the minimum frame count
    numeric_data = {}
    min_frames = float('inf')
    min_frames_key = None
    
    for group_name, field_name in numeric_fields:
        if group_name in ep and field_name in ep[group_name]:
            data = ep[group_name][field_name][:]
            key = f"{group_name}/{field_name}"
            numeric_data[key] = {
                'data': data[:, 1:],  # Data without timestamp column
                'ts': data[:, 0],      # Timestamp column
            }
            if data.shape[0] < min_frames:
                min_frames = data.shape[0]
                min_frames_key = key
    
    # Step 2: Load image timestamps and find minimum
    image_data = {}
    for img_field in image_fields:
        if img_field in ep:
            img_keys = sorted(ep[img_field].keys())
            # Extract timestamps from filenames (remove .jpg suffix)
            img_ts = np.array([int(k.replace('.jpg', '')) for k in img_keys])
            image_data[img_field] = {
                'keys': img_keys,
                'ts': img_ts,
            }
            if len(img_keys) < min_frames:
                min_frames = len(img_keys)
                min_frames_key = img_field
    print()
    print(f"Minimum frames: {min_frames} (from {min_frames_key})")
    
    # Step 3: Get the reference timestamps (from the field with minimum frames)
    if min_frames_key in numeric_data:
        ref_ts = numeric_data[min_frames_key]['ts']
    else:
        ref_ts = image_data[min_frames_key]['ts']
    
    # Step 4: Align all numeric data to the reference timestamps
    aligned_result = {}
    
    for key, item in numeric_data.items():
        data = item['data']
        ts = item['ts']
        
        if key == min_frames_key:
            # This is the reference, no need to align
            aligned_result[key] = data
        else:
            # Align to reference timestamps
            # array1 is the shorter ref (dummy), array2 is real data that gets trimmed
            # auto_match returns (array1, trimmed_array2), so we need the second value
            _, aligned_data = auto_match(
                np.column_stack([ref_ts, np.zeros((len(ref_ts), data.shape[1]))]),  # dummy ref data
                np.column_stack([ts, data]),  # actual data with ts
                ref_ts,
                ts
            )
            aligned_result[key] = aligned_data[:, 1:]  # Remove timestamp column
    
    # Step 5: Align all image data to the reference timestamps
    for img_field, item in image_data.items():
        img_keys = item['keys']
        img_ts = item['ts']
        
        if img_field == min_frames_key:
            # This is the reference, use all images
            aligned_images = [ep[img_field][k][:] for k in img_keys]
        else:
            # Find closest indices to reference timestamps
            idxs = np.array([np.abs(img_ts - t).argmin() for t in ref_ts])
            aligned_images = [ep[img_field][img_keys[i]][:] for i in idxs]
        
        aligned_result[img_field] = np.array(aligned_images)
    
    # print(f"Aligned data shapes:")
    # for key, data in aligned_result.items():
    #     print(f"  {key}: {data.shape}")

    original_img_frames = max(len(item['keys']) for item in image_data.values()) if image_data else 0
    print(f"原始数据中该 episode 帧数: {original_img_frames} -> 对齐后帧数: {min_frames}")
    
    return aligned_result


# ============================================================================
# MCAP 数据对齐函数 (用于 GenRobot 10Kh-RealOmin-OpenData)
# ============================================================================


def get_interp1d(t, x):
    """创建线性插值器"""
    return si.interp1d(
        t, x, 
        axis=0, bounds_error=False, 
        fill_value=(x[0], x[-1])
    )


class PoseInterpolator:
    """位姿插值器 (使用 SLERP 对四元数插值)"""
    def __init__(self, t, x):
        pos = x[:, :3]
        rot = st.Rotation.from_quat(x[:, 3:])
        self.pos_interp = get_interp1d(t, pos)
        self.rot_interp = st.Slerp(t, rot)
    
    @property
    def x(self):
        return self.pos_interp.x
    
    def __call__(self, t):
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)
        
        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        rvec = rot.as_quat()
        pose = np.concatenate([pos, rvec], axis=-1)
        return pose


def remove_duplicate_timestamps(t, y):
    """移除重复时间戳"""
    if len(t) != len(y):
        raise ValueError("t and y must have same length")
    
    _, unique_indices = np.unique(t, return_index=True)
    unique_indices = np.sort(unique_indices)
    
    return t[unique_indices], y[unique_indices]


def interpolate_topic_data(topic_data, ref_timestamps, inter_type="linear"):
    """
    将 topic 数据插值到参考时间戳
    
    Args:
        topic_data: McapLoader.get_topic_data() 返回的数据列表
        ref_timestamps: 参考时间戳数组
        inter_type: "linear" 或 "pose" (位姿用 SLERP)
    
    Returns:
        np.ndarray: 插值后的数据
    """
    assert inter_type in ["linear", "pose"]
    
    data_array = np.array([d["decode_data"] for d in topic_data])
    ts_array = np.array([d["data"].header.timestamp for d in topic_data])
    
    # 移除重复时间戳
    clean_ts, clean_data = remove_duplicate_timestamps(ts_array, data_array)
    
    if clean_data.shape[0] != data_array.shape[0]:
        print(f"  找到重复时间戳，原始: {data_array.shape[0]} -> 清理后: {clean_data.shape[0]}")
    
    assert clean_data.ndim == 2, f"Expected 2D array, got {clean_data.ndim}D"
    
    if inter_type == "linear":
        interp = get_interp1d(t=clean_ts, x=clean_data)
        result = interp(ref_timestamps).astype(np.float32)
    elif inter_type == "pose":
        interp = PoseInterpolator(t=clean_ts, x=clean_data)
        result = interp(ref_timestamps).astype(np.float32)
    
    return result


def align_mcap_data(bag, config):
    """
    将 MCAP 数据对齐到参考 topic 的时间戳 (通用版本)
    使用官方 das-datakit 的同步方法
    
    Args:
        bag: McapLoader 实例 (已加载数据)
        config: RobotConfig 实例，包含 mcap_ref_topic, mcap_camera_topics, mcap_numeric_topics
    
    Returns:
        dict: 包含对齐后的数据，key 为 lerobot 字段名
            - 相机数据: list of np.ndarray (N, H, W, C)
            - 数值数据: np.ndarray (N, dim)
            - "episode_length": int
    """
    import time
    t0 = time.time()
    
    ref_topic = config.mcap_ref_topic
    camera_topics = config.mcap_camera_topics
    numeric_topics = config.mcap_numeric_topics
    
    if not ref_topic:
        raise ValueError("config.mcap_ref_topic must be set")
    
    # 收集所有需要加载的 topics
    all_topics = [ref_topic]
    
    # 添加其他相机 topics (排除参考 topic，因为已经添加了)
    for field_name, topic in camera_topics.items():
        if topic not in all_topics:
            all_topics.append(topic)
    
    # 添加数值数据 topics
    for field_name, (topic, interp_type, shape) in numeric_topics.items():
        if topic not in all_topics:
            all_topics.append(topic)
    
    # 批量加载所有 topics
    bag.load_topics(all_topics, auto_sync=False)
    t1 = time.time()
    print(f"  [计时] load_topics ({len(all_topics)} topics): {t1-t0:.2f}s")
    
    # 获取参考 topic 数据
    ref_data = bag.get_topic_data(ref_topic)
    if ref_data is None or len(ref_data) == 0:
        raise ValueError(f"Reference topic {ref_topic} has no data")
    
    ref_timestamps = np.array([d["data"].header.timestamp for d in ref_data])
    ref_seq_nums = bag.get_topic_seq_num(ref_topic)
    episode_length = len(ref_data)
    print(f"  参考帧数 ({ref_topic}): {episode_length}")
    
    result = {"episode_length": episode_length}
    
    t2 = time.time()
    
    # ========================================================================
    # 处理相机数据 (使用官方 register_sync_relation_with_time 方法)
    # ========================================================================
    # 收集需要同步的非参考相机 topics
    sync_camera_topics = []
    for field_name, topic in camera_topics.items():
        if topic != ref_topic:
            sync_camera_topics.append(topic)
    
    # 注册同步关系
    for sync_topic in sync_camera_topics:
        success = bag.register_sync_relation_with_time(ref_topic, sync_topic)
        if not success:
            raise ValueError(f"相机 topic '{sync_topic}' 同步失败！请检查配置或 MCAP 文件。")
    
    t2_5 = time.time()
    print(f"  [计时] 注册同步关系: {t2_5-t2:.2f}s")
    
    # ========================================================================
    # 优化：预先构建 seq_num -> list_idx 的映射，避免重复查询
    # ========================================================================
    # 为每个相机 topic 构建 seq_num -> decode_data 的直接映射
    camera_seq2data = {}
    for topic in sync_camera_topics:
        topic_data = bag.get_topic_data(topic)
        seq2data = {}
        for item in topic_data:
            if hasattr(item["data"], "header"):
                seq = item["data"].header.sequence_num
                seq2data[seq] = item["decode_data"]
        camera_seq2data[topic] = seq2data
    
    # 参考相机也建立映射
    ref_seq2data = {}
    for item in ref_data:
        if hasattr(item["data"], "header"):
            seq = item["data"].header.sequence_num
            ref_seq2data[seq] = item["decode_data"]
    
    # 初始化相机数据容器
    camera_data = {field_name: [] for field_name in camera_topics.keys()}
    
    # 按 seq_num 获取同步的相机数据 (使用预建立的映射，避免重复查询)
    for seq_num in ref_seq_nums:
        for field_name, topic in camera_topics.items():
            if topic == ref_topic:
                # 参考相机直接取
                img = ref_seq2data.get(seq_num)
                if img is None:
                    raise ValueError(f"参考相机在 seq_num={seq_num} 无数据！")
                camera_data[field_name].append(img)
            else:
                # 非参考相机：从同步图获取对应的 seq_num
                sync_info = bag.sync_graph.get_relations(ref_topic, seq_num)
                sync_seq = sync_info.get(topic)
                if sync_seq is None:
                    raise ValueError(f"相机 '{field_name}' 对应的 topic '{topic}' 在 seq_num={seq_num} 无同步关系！")
                img = camera_seq2data[topic].get(sync_seq)
                if img is None:
                    raise ValueError(f"相机 '{field_name}' 对应的 topic '{topic}' 在 seq_num={sync_seq} 无数据！")
                camera_data[field_name].append(img)
    
    # 保存相机数据到结果
    for field_name in camera_topics.keys():
        result[field_name] = camera_data[field_name]
    
    t3 = time.time()
    print(f"  [计时] 相机数据处理: {t3-t2_5:.2f}s")
    
    # ========================================================================
    # 处理数值数据 (使用插值方法，与官方一致)
    # ========================================================================
    for field_name, (topic, interp_type, shape) in numeric_topics.items():
        topic_data = bag.get_topic_data(topic)
        
        if topic_data and len(topic_data) > 0:
            result[field_name] = interpolate_topic_data(topic_data, ref_timestamps, inter_type=interp_type)
        else:
            # 没有数据 - 直接报错
            raise ValueError(f"数值字段 '{field_name}' 对应的 topic '{topic}' 无数据！请检查配置或 MCAP 文件。")
    
    t4 = time.time()
    print(f"  [计时] 数值数据插值: {t4-t3:.2f}s")
    print(f"  [计时] align_mcap_data 总计: {t4-t0:.2f}s")
    
    return result
