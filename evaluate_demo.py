import motmetrics as mm
import numpy as np

# Create an accumulator that will be updated during each frame
acc = mm.MOTAccumulator(auto_id=True)

# Call update once for per frame. For now, assume distances between frame objects / hypotheses are given.
acc.update(
    [1, 2],  # Ground truth objects in this frame
    [1, 2, 3],  # Detector hypotheses in this frame
    [
        [0.1, np.nan, 0.3],  # Distances from object 1 to hypotheses 1, 2, 3
        [0.5, 0.2, 0.3]  # Distances from object 2 to hypotheses 1, 2, 3
    ]
)
# print(acc.events)  # a pandas DataFrame containing all events
# print('-' * 40)
# print(acc.mot_events)  # a pandas DataFrame containing MOT only events

frameid = acc.update(
    [1, 2],
    [1],
    [
        [0.2],
        [0.4]
    ]
)
print('-' * 40)
# print(acc.mot_events.loc[frameid])

gt_file = r"D:\dataset\MOT\MOT17\train\MOT17-02-DPM\gt\gt.txt"
"""  文件格式如下
1,0,1255,50,71,119,1,1,1
2,0,1254,51,71,119,1,1,1
3,0,1253,52,71,119,1,1,1
...
"""

ts_file = r"D:\dataset\MOT\MOT17\train\MOT17-02-DPM\gt\gt.txt"
"""  文件格式如下
1,1,1240.0,40.0,120.0,96.0,0.999998,-1,-1,-1
2,1,1237.0,43.0,119.0,96.0,0.999998,-1,-1,-1
3,1,1237.0,44.0,117.0,95.0,0.999998,-1,-1,-1
...
"""

gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)  # 读入GT
ts = mm.io.loadtxt(ts_file, fmt="mot15-2D")  # 读入自己生成的跟踪结果

acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)  # 根据GT和自己的结果，生成accumulator，distth是距离阈值
mh = mm.metrics.create()
# 打印单个accumulator
summary = mh.compute(acc,
                     metrics=['num_frames', 'mota', 'motp'],  # 一个list，里面装的是想打印的一些度量
                     name='acc')  # 起个名
# print(summary)


# 打印多个accumulators
summary = mh.compute_many([acc, acc.events.loc[0:1]],  # 多个accumulators组成的list
                          metrics=['num_frames', 'mota', 'motp'],
                          names=['full', 'part'])  # 起个名
# print(summary)

summary = mh.compute_many([acc, acc.events.loc[0:1]],
                          metrics=mm.metrics.motchallenge_metrics,
                          names=['full', 'part'])

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)
