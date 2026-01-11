import numpy as np
import h5py



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
