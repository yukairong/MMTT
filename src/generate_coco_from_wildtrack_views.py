import json
import os
import shutil

from utils.dataset_utils import parse_json_file, xyxy_convert_to_xywh

DATSET_SEQS_INFO = {
    "WildTrack": {"img_width": 1920, "img_height": 1080, "seq_length": 401}
}

def generate_coco_from_wildtrack_views(data_root=None, split_name=None,
                                       seqs_names=None, frame_range=None):
    if frame_range is None:
        frame_range = {'start': 0.0, 'end': 1.0}

    coco_dir = os.path.join(data_root, split_name)

    if os.path.isdir(coco_dir):
        shutil.rmtree(coco_dir)
    os.mkdir(coco_dir)

    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    annotations['categories'] = [
        {
            'supercategory': 'person',
            'name': 'person',
            'id': 1
        }
    ]
    annotations['annotations'] = []

    annotations_dir = os.path.join(os.path.join(data_root, 'annotations'))
    if not os.path.isdir(annotations_dir):
        os.mkdir(annotations_dir)

    # 标注信息文件路径名称
    annotation_dir_list = [os.path.join(annotations_dir, f'{seq_name}.json') for seq_name in seqs_names]

    # imgs
    img_id = 0
    imgs_data_root = os.path.join(data_root, 'Image_subsets')

    seqs = sorted(os.listdir(imgs_data_root))

    if seqs_names is not None:
        seqs = [s for s in seqs if s in seqs_names]


    annotations['sequences'] = seqs # 保存所有视角信息
    annotations['frame_range'] = frame_range

    # 对每个视角下的所有图片进行处理
    for view_id, seq in enumerate(seqs):
        img_width = DATSET_SEQS_INFO['WildTrack']['img_width']
        img_height = DATSET_SEQS_INFO['WildTrack']['img_height']
        seq_length = DATSET_SEQS_INFO['WildTrack']['seq_length']

        # 遍历所有该视角下的图片
        seq_list_dir = os.listdir(os.path.join(imgs_data_root, seq))

        start_frame = int(frame_range['start'] * seq_length)
        end_frame = int(frame_range['end'] * seq_length)

        # 取出起始帧到结束帧区间内的所有图片
        seq_list_dir = seq_list_dir[start_frame: end_frame]

        print(f"{seq}: {len(seq_list_dir)}/{seq_length}")
        seq_length = len(seq_list_dir)

        first_frame_image_id = -1   #记录第一帧的图片id
        for i, img in enumerate(sorted(seq_list_dir)):
            if i == 0:
                first_frame_image_id = img_id

            assert first_frame_image_id >= 0, "not found the first img's id"

            annotations['images'].append({
                'file_full_name': f'{seq}_{img}',
                'file_name': f'{img}',
                'height': img_height,
                'width': img_width,
                'id': img_id,
                'frame_id': i,
                'seq_length': seq_length,
                'first_frame_image_id': first_frame_image_id,
                'view_id': view_id
            })

            img_id += 1

            os.symlink(os.path.join(imgs_data_root, seq, img),
                       os.path.join(coco_dir, f"{seq}_{img}"))

    # 对标注信息进行处理
    img_file_name_to_id = {
        img_dict['file_full_name']: img_dict['id']
        for img_dict in annotations['images']
    }
    for seq_id, seq in enumerate(seqs):
        annotation_id = 0   # 每个视角下的标注id
        # wildtrack数据集的标注信息文件路径
        annotation_data_dir = os.path.join(data_root, "annotations_positions")

        if not os.path.exists(annotation_data_dir):
            print(f"{annotation_data_dir} path error!")

        # wildtrack数据集中所有Json标注文件列表
        annotation_data_list = os.listdir(annotation_data_dir)

        assert len(annotation_data_list) != 0, "annotation file is empty"

        seq_annotations = []
        seq_annotations_per_frame = {}

        # 遍历所有标注信息文件
        for seq_frame_id, annotation_data in enumerate(annotation_data_list):
            json_file = os.path.join(annotation_data_dir, annotation_data)

            if not os.path.isfile(json_file):
                print(f"{json_file} this json file is not exist, already pass")
                continue
            json_res = parse_json_file(json_file)

            # 对同一时刻,对每个视角下不同的人的标注信息进行遍历
            for res in json_res:
                personId = res['personID']
                positionId = res['positionID']
                view_annotation_data_list = res['views']

                for view_annotation_data in view_annotation_data_list:
                    view_id = view_annotation_data['viewNum']

                    # 只存储同一视角下的标注信息
                    if view_id != seq_id:
                        continue
                    xmax = view_annotation_data['xmax']
                    xmin = view_annotation_data['xmin']
                    ymax = view_annotation_data['ymax']
                    ymin = view_annotation_data['ymin']

                    bbox = list(xyxy_convert_to_xywh((xmin, ymin, xmax, ymax)))

                    area = bbox[2] * bbox[3]
                    visibility = 1 if area != 0 else 0
                    frame_id = seq_frame_id # 记录当前的序列帧id
                    image_id = img_file_name_to_id.get(f'{seq}_{annotation_data.replace(".json", ".png")}')
                    if image_id is None:
                        continue

                    track_id = personId

                    annotation = {
                        "id": annotation_id,
                        "bbox": bbox,
                        "image_id": image_id,
                        "segmentation": [],
                        "ignore": 0 if visibility == 1 else 1,
                        "area": area,
                        "iscrowd": 0,
                        "view_id": seq_id,
                        "seq": seq,
                        "category_id": annotations['categories'][0]['id'],
                        "track_id": personId
                    }

                    seq_annotations.append(annotation)

                    if frame_id not in seq_annotations_per_frame:
                        seq_annotations_per_frame[frame_id] = {}

                    if seq_id not in seq_annotations_per_frame[frame_id]:
                        seq_annotations_per_frame[frame_id][seq_id] = []

                    seq_annotations_per_frame[frame_id][seq_id].append(annotation)

                    annotation_id += 1
        seq_annotations = sorted(seq_annotations, key=lambda x: x['id'], reverse=False)

        annotations['annotations'].extend(seq_annotations)
        with open(annotation_dir_list[seq_id], 'w') as anno_file:
            json.dump(annotations, anno_file, indent=4)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Generate COCO from WildTrack")
    data_root = r"D:\datasets\Wildtrack_dataset_full\Wildtrack_dataset"
    seqs_names = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    generate_coco_from_wildtrack_views(
        data_root=data_root, split_name="wildtrack_train_coco",
        seqs_names=seqs_names, frame_range=None
    )