import shutil
import os

parser = argparse.ArgumentParser(description='Prepare the training data.')
parser.add_argument('--augment', default=True, help='Data augmentation indicator.')
parser.add_argument('--testSubj', type=str, default='R7', help='The test subject that never involved while training and'
                                                               ' evaluation.')
parser.add_argument('--Transition', default=True, help='Including transition data as the third class.')

args = parser.parse_args()

os.makedirs('./data')

# 源文件夹路径
if args.augment:
    augment = 'aug_results'
else:
    augment = 'noAug_results'
src_folder = os.path.join("../RawData/data for classification/", augment, args.testSubj)
# 目标文件夹路径
dest_folder = "./data/testFold"
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)
    os.makedirs(os.path.join(dest_folder, 'video_data'))

files = os.listdir(src_folder)
# 遍历文件列表并筛选出子文件夹
actionList = [f for f in files if os.path.isdir(os.path.join(src_folder, f))]
for action in actionList:
    if (not args.Transition) and action == 'Transition':
        continue
    src_folder_temp = os.path.join(src_folder, action)
    if not os.path.exists(src_folder_temp):
        os.makedirs(src_folder_temp)
    dest_folder_temp = os.path.join(os.path.join(dest_folder, 'video_data'), action)
    if not os.path.exists(dest_folder_temp):
        os.makedirs(dest_folder_temp)
    # 获取源文件夹中的文件列表
    files = os.listdir(src_folder_temp)
    # 遍历文件列表并将每个文件复制到目标文件夹中
    for file in files:
        src_path = os.path.join(src_folder_temp, file)
        dest_path = os.path.join(dest_folder_temp, args.testSubj + '_' + file)
        shutil.copy(src_path, dest_path)


# 使用os模块的listdir函数获取文件夹中的所有子文件夹和文件
folder_path = os.path.join("../RawData/data for classification/", augment)
files = os.listdir(folder_path)
# 遍历文件列表并筛选出子文件夹
subjList = [f for f in files if os.path.isdir(os.path.join(folder_path, f))]

dest_folder = "./data/trainFold"
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)
    os.makedirs(os.path.join(dest_folder, 'video_data'))
# 目标文件夹路径
dest_folder = os.path.join(dest_folder, 'video_data')
for subj in subjList:
    # 源文件夹路径
    src_folder = os.path.join("../RawData/data for classification/", augment, subj)
    files = os.listdir(src_folder)
    actionList = [f for f in files if os.path.isdir(os.path.join(src_folder, f))]
    for action in actionList:
        if (not args.Transition) and action == 'Transition':
            continue
        src_folder_temp = os.path.join(src_folder, action)
        if not os.path.exists(src_folder_temp):
            os.makedirs(src_folder_temp)
        dest_folder_temp = os.path.join(dest_folder, action)
        if not os.path.exists(dest_folder_temp):
            os.makedirs(dest_folder_temp)
        # 获取源文件夹中的文件列表
        files = os.listdir(src_folder_temp)
        # 遍历文件列表并将每个文件复制到目标文件夹中
        for file in files:
            src_path = os.path.join(src_folder_temp, file)
            dest_path = os.path.join(dest_folder_temp, subj + '_' + file)
            shutil.copy(src_path, dest_path)