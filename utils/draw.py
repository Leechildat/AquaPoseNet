import math
import os
import torch
import time
import matplotlib.pyplot as plt


def get_acc(f: list, split_ema=True):
    if isinstance(f, str):
        f = open(f, "r").readlines()

    emaaccs = None
    accs = []
    for i, line in enumerate(f):
        # if "Epoch:" in line:
        #     l: str = line.split("Accuracy ")[-1].split(" (")[-1].split(")")[0]
        #     accs.append(dict(miou=float(l)))

        if "| 256x256" in line:
            l: str = line.split(" | ")[9].split(" |\n")[0]
            # l: str = line.split(" | ")[-1].split(" (")[-1].split(")")[0]
            # l: str = line.split(" | ")[-1].split(" (")[-1].split(")")[0]
            # l: str = line.split(" | ")[-1].split(" (")[-1].split(")")[0]
            accs.append(dict(miou=float(l)))

    accs = accs[:30]
    accs = dict(miou=[a['miou'] for a in accs])
    x_axis = range(len(accs['miou']))
    return x_axis, accs


def get_loss(f: list, x1e=torch.tensor(list(range(0, 1253, 10))).view(1, -1) / 1253, scale=1):
    if isinstance(f, str):
        f = open(f, "r").readlines()

    losses = []
    for i, line in enumerate(f):
        if "Epoch:" in line:
            l = line.split("Loss ")[-1].split(" (")[1].split(")")[0]
            # losses.append(float(l))
            losses.append(math.sqrt(float(l)))

    x = x1e
    x = x.repeat(len(losses) // x.shape[1] + 1, 1)
    x = x + torch.arange(0, x.shape[0]).view(-1, 1)
    x = x.flatten().tolist()
    x_axis = x[:len(losses)]

    losses = [l * scale for l in losses]

    return x_axis, losses


def draw_fig(data: list, title="", xlabel="", xlim=(0, 301), ylim=(68, 84), xstep=None, ystep=None, save_path="./show.jpg"):
    assert isinstance(data[0], dict)
    from matplotlib import pyplot as plot
    fig, ax = plot.subplots(dpi=300, figsize=(13, 8))
    for d in data:
        length = min(len(d['x']), len(d['y']))
        x_axis = d['x'][:length]
        y_axis = d['y'][:length]
        label = d['label']
        ax.plot(x_axis, y_axis, label=label)
    plot.xlim(xlim)
    plot.ylim(ylim)
    plot.legend(fontsize=20)
    ax.set_title(title, fontsize=24)
    ax.set_xlabel(xlabel, fontsize=20)
    if xstep is not None:
        plot.xticks(torch.arange(xlim[0], xlim[1], xstep).tolist())
    if ystep is not None:
        plot.yticks(torch.arange(ylim[0], ylim[1], ystep).tolist())
    plot.grid()
    # plot.show()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot.savefig(save_path)


def main_vssm():
    showpath = os.path.join(os.path.dirname(__file__), "./show/log")


    file_list = {'Ours(x2)': "../output/ModelFishPose/LiteHRNet_50/model_256x256_VMHRNet18_2222_tiny/model_256x256_VMHRNet18_2222_tiny_2024-03-27-15-28_train.log",
                 'Ours(x4)': "../output/ModelFishPose/LiteHRNet_50/model_256x256_VMHRNet18_2242_tiny/model_256x256_VMHRNet18_2242_tiny_2024-03-27-11-22_train.log",
                 'Ours(x9)': "../output/ModelFishPose/LiteHRNet_50/model_256x256_VMHRNet18_2292_tiny/model_256x256_VMHRNet18_2292_tiny_2024-03-27-13-38_train.log",
                 'Ours(x15)': "../output/ModelFishPose/LiteHRNet_50/model_256x256_VMHRNet18_22F2_small/model_256x256_VMHRNet18_22F2_small_2024-03-28-06-26_train.log",
                 'Hourglass(x4)': "../output/ModelFishPose/HgNet_50/model_256×256_4stack_hourglass/model_256×256_4stack_hourglass_2024-03-27-19-16_train.log",
                 'Hourglass(x2)': "../output/ModelFishPose/HgNet_50/model_256×256_2stack_hourglass/model_256×256_2stack_hourglass_2024-03-27-17-49_train.log",
                 'ResNet50': "../output/ModelFishPose/PoseResNet_50/model_256×256_resnet50/model_256×256_resnet50_2024-03-28-00-36_train.log",
                 'ResNet101': "../output/ModelFishPose/PoseResNet_101/model_256×256_resnet101/model_256×256_resnet101_2024-03-28-04-36_train.log",
                 'LiteHRNet-18': "../output/ModelFishPose/LiteHRNet_50/model_256×256_LiteHRNet18/model_256×256_LiteHRNet18_2024-03-29-16-44_train.log",
                 'LiteHRNet-30': "../output/ModelFishPose/LiteHRNet_50/model_256×256_LiteHRNet30/model_256×256_LiteHRNet30_2024-03-29-17-42_train.log"
                 }
    # vssmdtiny_our = "../output/ModelFishPose/LiteHRNet_50/model_256x256_VMHRNet18_22F2_small/model_256x256_VMHRNet18_22F2_small_2024-03-28-06-26_train.log"
    # vssmdtiny_hourglass = "../output/ModelFishPose/HgNet_50/model_256×256_4stack_hourglass/model_256×256_4stack_hourglass_2024-03-27-19-16_train.log"
    # vssmdtiny_resnet50 = "..output/ModelFishPose/PoseResNet_50/model_256×256_resnet50/model_256×256_resnet50_2024-03-28-00-36_train.log"
    # vssmdtiny_LiteHR = "../output/ModelFishPose/LiteHRNet_50/model_256×256_LiteHRNet18/model_256×256_LiteHRNet18_2024-03-29-16-44_train.log"

    num_list = {'Ours(x15)': 39,
                 'Hourglass(x4)': 13,
                 'ResNet50': 13,
                 'LiteHRNet-18': 4
                 }

    acc_list = []
    loss_list = []

    for name, path in file_list.items():
        x, accs = get_acc(path, split_ema=False)
        print(f"{name}, Max: {max(accs['miou'])} epoch: {accs['miou'].index(max(accs['miou'])) + 1}")
        # print(f"accs: {accs['acc1'][-1]}")
        # print(f"emaaccs: {emaaccs['acc1'][-1]}")
        # lx, losses = get_loss(path, x1e=torch.tensor(list(range(0, num_list[name], 10))).view(1, -1) / num_list[name], scale=1)
        # vssmdtiny = dict(xaxis=x, accs=accs, loss_xaxis=lx, losses=losses, name=name)
        # acc_list.append(dict(x=vssmdtiny['xaxis'], y=vssmdtiny['accs']['miou'], label=name))
        # loss_list.append(dict(x=vssmdtiny['loss_xaxis'], y=vssmdtiny['losses'], label=name))

    # if True:
    #     title = ("Train Accuracy")
    #     xlabel = ("Epochs")
    #     draw_fig(data=acc_list, title=title, xlabel=xlabel, xlim=(0, 100), ylim=(0, 1), xstep=20, ystep=0.05, save_path=f"{showpath}/ap.jpg")
    #
    # if True:
    #     title = ("Train Loss")
    #     xlabel = ("Epochs")
    #     draw_fig(data=loss_list, title=title, xlabel=xlabel, xlim=(0, 100), ylim=(0, 0.2), xstep=20, ystep=0.02, save_path=f"{showpath}/loss.jpg")

main_vssm()