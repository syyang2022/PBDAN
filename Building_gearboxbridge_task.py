import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os

# 创建保存目录
save_dir = "./gearset/datagearset_thirdspace/"
os.makedirs(save_dir, exist_ok=True)


# ## Define the FFT function
def hua_fft(data, fs):
    n = len(data)  # length of the signal
    k = np.arange(n) / n
    frq = fs * k  # two sides frequency range
    frq = frq[range(n // 2)]  # one side frequency range
    Y = np.fft.fft(data) / n  # fft computing and normalization
    Y = Y[range(n // 2)]
    return [frq, abs(Y)]


def FFT_sample(data, sample_freq):
    FFT = []
    for i in range(data.shape[0]):
        fft = hua_fft(data[i, :], sample_freq)
        FFT.append(fft)
    return np.array(FFT)


def image(File_num, size, File_path, Save_path):
    P = []
    for i in range(File_num):
        file_name = File_path + "_%d.jpg" % (i)
        pic = Image.open(file_name)
        pic = pic.resize((size, size))
        pic = np.array(pic)
        P.append(pic)
    np.save(Save_path + "FFT.npy", P)


def transformer(data, File_path):
    for i in range(data.shape[0]):
        plt.figure(figsize=(2, 2), dpi=300)
        plt.axis('off')
        plt.plot(data[i, 0, 128:1024], data[i, 1, 128:1024], 'r')
        plt.savefig(File_path + "_%d.jpg" % (i))
        plt.close()


# ## Define the dataset reading function
def work_data(frame_size, step, data_size, path_list, channel):
    helical1, helical2, helical3, helical4, helical5, helical6, = [], [], [], [], [], []
    spur1, spur2, spur3, spur4, spur5, spur6, spur7, spur8 = [], [], [], [], [], [], [], []
    work1 = np.loadtxt('./gearset/raw/' + path_list[0])[:, channel]
    work2 = np.loadtxt('./gearset/raw/' + path_list[1])[:, channel]
    work3 = np.loadtxt('./gearset/raw/' + path_list[2])[:, channel]
    work4 = np.loadtxt('./gearset/raw/' + path_list[3])[:, channel]
    work5 = np.loadtxt('./gearset/raw/' + path_list[4])[:, channel]
    work6 = np.loadtxt('./gearset/raw/' + path_list[5])[:, channel]
    work7 = np.loadtxt('./gearset/raw/' + path_list[6])[:, channel]
    work8 = np.loadtxt('./gearset/raw/' + path_list[7])[:, channel]
    work9 = np.loadtxt('./gearset/raw/' + path_list[8])[:, channel]
    work10 = np.loadtxt('./gearset/raw/' + path_list[9])[:, channel]
    work11 = np.loadtxt('./gearset/raw/' + path_list[10])[:, channel]
    work12 = np.loadtxt('./gearset/raw/' + path_list[11])[:, channel]
    work13 = np.loadtxt('./gearset/raw/' + path_list[12])[:, channel]
    work14 = np.loadtxt('./gearset/raw/' + path_list[13])[:, channel]

    for i in range(data_size):
        helical1.append(work1[i * step: i * step + frame_size].tolist())
        helical2.append(work2[i * step: i * step + frame_size].tolist())
        helical3.append(work3[i * step: i * step + frame_size].tolist())
        helical4.append(work4[i * step: i * step + frame_size].tolist())
        helical5.append(work5[i * step: i * step + frame_size].tolist())
        helical6.append(work6[i * step: i * step + frame_size].tolist())
        spur1.append(work7[i * step: i * step + frame_size].tolist())
        spur2.append(work8[i * step: i * step + frame_size].tolist())
        spur3.append(work9[i * step: i * step + frame_size].tolist())
        spur4.append(work10[i * step: i * step + frame_size].tolist())
        spur5.append(work11[i * step: i * step + frame_size].tolist())
        spur6.append(work12[i * step: i * step + frame_size].tolist())
        spur7.append(work13[i * step: i * step + frame_size].tolist())
        spur8.append(work14[i * step: i * step + frame_size].tolist())

    data = np.concatenate((np.array(helical1), np.array(helical2),
                           np.array(helical3), np.array(helical4),
                           np.array(helical5), np.array(helical6),
                           np.array(spur1), np.array(spur2),
                           np.array(spur3), np.array(spur4),
                           np.array(spur5), np.array(spur6),
                           np.array(spur7), np.array(spur8)), axis=0)
    labels = np.zeros((data_size * 14, 14))
    for i in range(14):
        labels[data_size * i:data_size + i * data_size, i] = 1

    return data, labels


# ## 新增：桥梁域创建函数
def create_bridge_domain(source_data, target_data, bridge_ratio):
    """
    创建桥梁域 - 作为源域和目标域的中间过渡域
    bridge_ratio: 桥梁域中源域数据的比例
    """
    # 确保两个域的数据量相同
    min_samples = min(len(source_data), len(target_data))
    source_samples = int(min_samples * bridge_ratio)
    target_samples = min_samples - source_samples

    # 从源域和目标域各取一部分数据构建桥梁域
    bridge_source = source_data[:source_samples]
    bridge_target = target_data[:target_samples]
    bridge_domain = np.vstack([bridge_source, bridge_target])

    return bridge_domain


def create_bridge_labels(source_label, target_label, bridge_samples):
    """
    创建桥梁域的标签
    桥梁域包含源域和目标域的部分数据，标签需要相应处理
    """
    source_samples = bridge_samples // 2
    target_samples = bridge_samples - source_samples

    bridge_label = np.vstack([
        source_label[:source_samples],
        target_label[:target_samples]
    ])

    return bridge_label


# ## Load the gear dataset
working_conditions = ["30hz", "35hz", "40hz", "45hz", "50hz"]
speed = ["High", "Low"]
i = 4
j = 2
m = 0
n = 0

source_data_path = [
    'helical 1/helical 1_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
    'helical 2/helical 2_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
    'helical 3/helical 3_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
    'helical 4/helical 4_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
    'helical 5/helical 5_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
    'helical 6/helical 6_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
    'spur 1/spur 1_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
    'spur 2/spur 2_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
    'spur 3/spur 3_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
    'spur 4/spur 4_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
    'spur 5/spur 5_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
    'spur 6/spur 6_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
    'spur 7/spur 7_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
    'spur 8/spur 8_' + working_conditions[i] + '_' + speed[m] + '_1.txt',
]

target_data_path = [
    'helical 1/helical 1_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
    'helical 2/helical 2_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
    'helical 3/helical 3_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
    'helical 4/helical 4_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
    'helical 5/helical 5_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
    'helical 6/helical 6_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
    'spur 1/spur 1_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
    'spur 2/spur 2_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
    'spur 3/spur 3_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
    'spur 4/spur 4_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
    'spur 5/spur 5_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
    'spur 6/spur 6_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
    'spur 7/spur 7_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
    'spur 8/spur 8_' + working_conditions[j] + '_' + speed[n] + '_1.txt',
]

frame_size = 6000
step = 25
data_size = 100
class_num = 6
batch_size = 128
save = True
split = 0.2
channel1 = 0
channel2 = 0

# load data
source_data, source_labels = work_data(frame_size, step, data_size, source_data_path, channel1)
target_data, target_labels = work_data(frame_size, step, data_size, target_data_path, channel2)
FFT1 = FFT_sample(source_data, 60000)
FFT2 = FFT_sample(target_data, 60000)

# ## Visualize the FFT spectrum of gear data
fig, axes = plt.subplots(4, 7, figsize=(28, 7), dpi=400)
start = 128
end = 1024
axes[0, 0].plot(FFT1[0, 0, start:end], FFT1[0, 1, start:end])
axes[0, 1].plot(FFT1[100, 0, start:end], FFT1[100, 1, start:end])
axes[0, 2].plot(FFT1[200, 0, start:end], FFT1[200, 1, start:end])
axes[0, 3].plot(FFT1[300, 0, start:end], FFT1[300, 1, start:end])
axes[0, 4].plot(FFT1[400, 0, start:end], FFT1[400, 1, start:end])
axes[0, 5].plot(FFT1[500, 0, start:end], FFT1[500, 1, start:end])
axes[0, 6].plot(FFT1[600, 0, start:end], FFT1[600, 1, start:end])

axes[1, 0].plot(FFT1[700, 0, start:end], FFT1[700, 1, start:end])
axes[1, 1].plot(FFT1[800, 0, start:end], FFT1[800, 1, start:end])
axes[1, 2].plot(FFT1[900, 0, start:end], FFT1[900, 1, start:end])
axes[1, 3].plot(FFT1[1000, 0, start:end], FFT1[1000, 1, start:end])
axes[1, 4].plot(FFT1[1100, 0, start:end], FFT1[1100, 1, start:end])
axes[1, 5].plot(FFT1[1200, 0, start:end], FFT1[1200, 1, start:end])
axes[1, 6].plot(FFT1[1300, 0, start:end], FFT1[1300, 1, start:end])

axes[2, 0].plot(FFT2[0, 0, start:end], FFT2[0, 1, start:end])
axes[2, 1].plot(FFT2[100, 0, start:end], FFT2[100, 1, start:end])
axes[2, 2].plot(FFT2[200, 0, start:end], FFT2[200, 1, start:end])
axes[2, 3].plot(FFT2[300, 0, start:end], FFT2[300, 1, start:end])
axes[2, 4].plot(FFT2[400, 0, start:end], FFT2[400, 1, start:end])
axes[2, 5].plot(FFT2[500, 0, start:end], FFT2[500, 1, start:end])
axes[2, 6].plot(FFT2[600, 0, start:end], FFT2[600, 1, start:end])

axes[3, 0].plot(FFT2[700, 0, start:end], FFT2[700, 1, start:end])
axes[3, 1].plot(FFT2[800, 0, start:end], FFT2[800, 1, start:end])
axes[3, 2].plot(FFT2[900, 0, start:end], FFT2[900, 1, start:end])
axes[3, 3].plot(FFT2[1000, 0, start:end], FFT2[1000, 1, start:end])
axes[3, 4].plot(FFT2[1100, 0, start:end], FFT2[1100, 1, start:end])
axes[3, 5].plot(FFT2[1200, 0, start:end], FFT2[1200, 1, start:end])
axes[3, 6].plot(FFT2[1300, 0, start:end], FFT2[1300, 1, start:end])

# ## Save the FFT spectrum
File_path1 = "./gearset/data/images/source/"
transformer(FFT1, File_path1)

File_path2 = "./gearset/data/images/target/"
transformer(FFT2, File_path2)
File_num = 1400
size = 64
Save_path1 = "./gearset/data/Source_"
Save_path2 = "./gearset/data/Target_"
image(File_num, size, File_path1, Save_path1)
image(File_num, size, File_path2, Save_path2)
np.save("./gearset/data/label_image.npy", source_labels)

# ## Build the OSDT task with Bridge Domains
F_0 = np.load("./gearset/data/Source_FFT.npy")
F_1 = np.load("./gearset/data/Target_FFT.npy")
X_0 = np.vstack((F_0[0:100], F_0[200:400], F_0[700:800], F_0[1000:1100], F_0[1200:1300]))
X_1 = np.vstack((F_1[0:100], F_1[200:400], F_1[700:800], F_1[1000:1100], F_1[1200:1300]))


def T0(X_0, X_1, out_num):
    if out_num == 1:
        # Transfer 1,3,4---1,3,4,6
        source = np.vstack((F_0[0:100], F_0[700:700], F_0[700:800], F_0[300:400], F_0[300:400]))
        target = np.vstack((F_1[0:100], F_1[700:800], F_1[300:400], F_1[1300:1400]))

        # 创建桥梁域
        bridge = create_bridge_domain(source, target, bridge_ratio=0.5)

        label_source = np.zeros((400, 4))
        label_source[0:100, 0] = 1
        label_source[100:200, 1] = 1
        label_source[200:300, 2] = 1
        label_source[300:400, 2] = 1

        label_target = np.zeros((400, 4))
        label_target[0:100, 0] = 1
        label_target[100:200, 1] = 1
        label_target[200:300, 2] = 1
        label_target[300:400, 3] = 1

        # 创建桥梁域的标签
        label_bridge = create_bridge_labels(label_source, label_target, len(bridge))

    if out_num == 2:
        # Transfer 1,3,4---1,3,4,5,6
        source = np.vstack((X_0[0:50], X_0[300:400], X_0[350:400], X_0[200:300], X_0[200:300], X_0[200:300]))
        target = np.vstack((X_1[0:100], X_1[300:400], X_1[200:300], F_1[1300:1400], X_1[500:600]))

        # 创建桥梁域
        bridge = create_bridge_domain(source, target, bridge_ratio=0.5)

        label_source = np.zeros((500, 4))
        label_source[0:50, 0] = 1
        label_source[50:200, 1] = 1
        label_source[200:300, 2] = 1
        label_source[300:500, 2] = 1

        label_target = np.zeros((500, 4))
        label_target[0:100, 0] = 1
        label_target[100:200, 1] = 1
        label_target[200:300, 2] = 1
        label_target[300:500, 3] = 1

        # 创建桥梁域的标签
        label_bridge = create_bridge_labels(label_source, label_target, len(bridge))

    if out_num == 3:
        # Transfer 1,3,4---1,2,3,4,5,6
        source = np.vstack((X_0[0:100], X_0[300:400], X_0[200:300], X_0[200:300], X_0[200:300], X_0[300:400]))
        target = np.vstack(
            (X_1[0:100], X_1[300:400], X_1[200:300], X_1[500:600], F_1[1300:1400], F_1[1300:1350], F_1[100:150]))

        # 创建桥梁域
        bridge = create_bridge_domain(source, target, bridge_ratio=0.5)

        label_source = np.zeros((600, 4))
        label_source[0:100, 0] = 1
        label_source[100:200, 1] = 1
        label_source[200:300, 2] = 1
        label_source[300:500, 2] = 1
        label_source[500:600, 1] = 1

        label_target = np.zeros((600, 4))
        label_target[0:100, 0] = 1
        label_target[100:200, 1] = 1
        label_target[200:300, 2] = 1
        label_target[300:600, 3] = 1

        # 创建桥梁域的标签
        label_bridge = create_bridge_labels(label_source, label_target, len(bridge))

    # 保存所有域的数据和标签
    np.save(f"{save_dir}T{out_num}_source_image.npy", source)
    np.save(f"{save_dir}T{out_num}_target_image.npy", target)
    np.save(f"{save_dir}T{out_num}_bridge_image.npy", bridge)
    np.save(f"{save_dir}T{out_num}_source_label.npy", label_source)
    np.save(f"{save_dir}T{out_num}_target_label.npy", label_target)
    np.save(f"{save_dir}T{out_num}_bridge_label.npy", label_bridge)


# Generate all tasks with bridge domains
print("Generating Task 1 with bridge domains...")
T0(X_0, X_1, 1)
print("Task 1 generation completed!")

print("Generating Task 2 with bridge domains...")
T0(X_0, X_1, 2)
print("Task 2 generation completed!")

print("Generating Task 3 with bridge domains...")
T0(X_0, X_1, 3)
print("Task 3 generation completed!")

print("All enhanced OSDT tasks with bridge domains have been generated successfully!")
print(f"Data saved to: {save_dir}")

# 保存目录（与之前一致）
save_dir = "./gearset/datagearset_thirdspace/"

tasks = ['Task 1', 'Task 2', 'Task 3']
categories = ['N', 'BF&SI', 'G&E', 'G']

# 用于存储每个任务的源域和目标域类别数量
source_counts = {}
target_counts = {}

for task_num in [1, 2, 3]:
    # 加载源域和目标域标签
    source_label = np.load(os.path.join(save_dir, f"T{task_num}_source_label.npy"))
    target_label = np.load(os.path.join(save_dir, f"T{task_num}_target_label.npy"))

    # 统计每个类别的样本数（对列求和，因为 one-hot 每行只有一个 1）
    source_counts[f"Task {task_num}"] = np.sum(source_label, axis=0).astype(int).tolist()
    target_counts[f"Task {task_num}"] = np.sum(target_label, axis=0).astype(int).tolist()

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, task in enumerate(tasks):
    x = np.arange(len(categories))
    width = 0.35

    axes[i].bar(x - width / 2, source_counts[task], width, label='Source', color='steelblue')
    axes[i].bar(x + width / 2, target_counts[task], width, label='Target', color='lightcoral')
    # 添加虚线网格
    axes[i].grid(True, linestyle='--', alpha=0.6)

    axes[i].set_title(task)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(categories)
    axes[i].tick_params(axis='y', labelleft=True)
    # 只在最左边的子图设置y轴标签
    if i == 0:
        axes[i].set_ylabel('Number of samples')
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'domain_category_counts.png'), dpi=300)

print("Bar chart of category counts saved successfully!")