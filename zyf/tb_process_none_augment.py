import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def parse_tensorboard_log(path):
    ea = EventAccumulator(path)
    ea.Reload()
    return {tag: pd.DataFrame(ea.Scalars(tag)) for tag in ea.Tags()['scalars']}

# 使用示例
import os
events=os.listdir("./none_augment")

all_loss=[]

for it in events:
    if it=='events.out.tfevents.1750394966.autodl-container-dbce458354-61007839':
        continue
    log_data = parse_tensorboard_log(f"./none_augment/{it}")
    # print(log_data)
    # print(log_data['train/loss_ctc'])  # 假设你有一个名为val_psnr的标量

    df=log_data['epoch/loss_att']
    print(df)

    df=log_data['epoch/acc']
    print(log_data["epoch/acc"])
    # 确定每个epoch的步数
    n_epochs = 2  # 假设包含10个epoch
    steps_per_epoch = len(df) // n_epochs

    # 将dataframe按照每个epoch的步数进行切分
    slices = np.array_split(df, n_epochs)

    # 对每个切片计算平均loss值
    average_losses = [slice['value'].mean() for slice in slices]

    # 将每个切片的平均loss值存储到一个列表中
    print(average_losses)

    all_loss+=average_losses[:2]

print(all_loss)