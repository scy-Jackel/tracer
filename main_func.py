# coding=utf-8
import os
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader


from config_tracer import config_tracer
from resnet import ResNet18
from model_cover import COVER_NET
from mydataset import MyDataset
import util
import numpy as np
# import cv2
import shutil
import torch.nn.functional as F
import math
import random
from PIL import Image


def train():
    class_net = ResNet18()
    class_net.cuda().train()

    cover_net = COVER_NET()
    cover_net.cuda().train()

    ckpt_cover_path = "./ckpt_cover/"
    data_path, step = util.load_weight(ckpt_cover_path)
    if step:
        cover_net.load_state_dict(torch.load(data_path))

    ckpt_class_path = "./ckpt_class/"
    data_path, step = util.load_weight(ckpt_class_path)
    if step:
        class_net.load_state_dict(torch.load(data_path))

    batch = 64
    lr = 7e-5
    weight_decay = 0

    optimizer = torch.optim.Adam(util.models_parameters([cover_net, class_net]), lr, weight_decay=weight_decay)
    ckpt_optimizer_path = "./optimizer_ckpt/"
    # data_path, step_optimizer = util.load_weight(ckpt_optimizer_path)
    # if step_optimizer != step:
    #     print 'optimizer step error'
    #     return
    # else:
    #     if step_optimizer:
    #         optimizer.load_state_dict(torch.load(data_path))

    data_path = '/home/cuiyang/data/CT_2d_lidc/train/'
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor(), ])
    train_set = datasets.ImageFolder(data_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=2)

    dataiter = iter(train_loader)
    lenth_iter = len(dataiter)
    read_num = 0

    height = 64
    width = 64
    torchHorizontal = torch.linspace(-width / 2, width / 2, width).cuda().view(1, 1, 1, width).expand(1, 1, height,
                                                                                                      width)
    torchVertical = torch.linspace(-height / 2, -height / 2, height).cuda().view(1, 1, height, 1).expand(1, 1, height,
                                                                                                         width)
    grid = torchHorizontal * torchHorizontal + torchVertical * torchVertical
    grid_max = torch.max(grid)
    grid = grid / grid_max

    step0 = 100000001

    start = step
    for step in range(start + 1, step0):
        print("step0-", step)
        images, labels = next(dataiter)
        read_num += 1
        if read_num == lenth_iter - 1:
            dataiter = iter(train_loader)
            read_num = 0
        images = images.cuda()
        labels = labels.float().cuda()

        batch_one, _, _, _ = images.size()
        grid_one = grid.expand(batch_one, 1, height, width)

        cover_mask = cover_net(images)

        class_out = class_net(images, cover_mask)
        class_out = torch.mean(class_out, dim=1)

        loss_class = (class_out - labels)
        loss_class = (loss_class * loss_class).mean()
        loss_cover = torch.mean(cover_mask)
        loss = loss_class + loss_cover * 0.5

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_class_c = loss_class.cpu().data.numpy()
        loss_cover_c = loss_cover.cpu().data.numpy()
        print("            loss_class:", loss_class_c, "   loss_cover:", loss_cover_c)

        if step % 5000 == 0:
            print("save weight at step:%d" % (step))
            util.save_weight(class_net, step, ckpt_class_path)
            print("save weight at step:%d" % (step))
            util.save_weight(cover_net, step, ckpt_cover_path)
            print("save optimizer weight at step:%d" % (step))
            util.save_weight(optimizer, step, ckpt_optimizer_path)


def train_for_pcl(data_url, save_url, model_url):
    class_net = ResNet18()
    class_net.cuda().train()

    cover_net = COVER_NET()
    cover_net.cuda().train()

    ckpt_cover_path = os.path.join(model_url, 'ckpt_cover')
    data_path, step = util.load_weight(ckpt_cover_path)
    if step:
        cover_net.load_state_dict(torch.load(data_path))

    ckpt_class_path = os.path.join(model_url, 'ckpt_class')
    data_path, step = util.load_weight(ckpt_class_path)
    if step:
        class_net.load_state_dict(torch.load(data_path))

    batch = 64
    lr = 7e-5
    weight_decay = 0

    optimizer = torch.optim.Adam(util.models_parameters([cover_net, class_net]), lr, weight_decay=weight_decay)

    ckpt_optimizer_path = os.path.join(model_url, 'optimizer_ckpt')
    data_path, step_optimizer = util.load_weight(ckpt_optimizer_path)
    if step_optimizer != step:
        print ('optimizer step error')
        return
    else:
        if step_optimizer:
            optimizer.load_state_dict(torch.load(data_path))

    data_path = data_url
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor(), ])
    train_set = datasets.ImageFolder(data_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=2)

    dataiter = iter(train_loader)
    lenth_iter = len(dataiter)
    read_num = 0

    height = 64
    width = 64
    torchHorizontal = torch.linspace(-width / 2, width / 2, width).cuda().view(1, 1, 1, width).expand(1, 1, height,
                                                                                                      width)
    torchVertical = torch.linspace(-height / 2, -height / 2, height).cuda().view(1, 1, height, 1).expand(1, 1, height,
                                                                                                         width)
    grid = torchHorizontal * torchHorizontal + torchVertical * torchVertical
    grid_max = torch.max(grid)
    grid = grid / grid_max

    step0 = 100000001

    start = step
    for step in range(start + 1, step0):
        print("step0-", step)
        images, labels = next(dataiter)
        read_num += 1
        if read_num == lenth_iter - 1:
            dataiter = iter(train_loader)
            read_num = 0
        images = images.cuda()
        labels = labels.float().cuda()

        batch_one, _, _, _ = images.size()
        grid_one = grid.expand(batch_one, 1, height, width)

        cover_mask = cover_net(images)

        class_out = class_net(images, cover_mask)
        class_out = torch.mean(class_out, dim=1)

        loss_class = (class_out - labels)
        loss_class = (loss_class * loss_class).mean()
        loss_cover = torch.mean(cover_mask)
        loss = loss_class + loss_cover * 0.5

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_class_c = loss_class.cpu().data.numpy()
        loss_cover_c = loss_cover.cpu().data.numpy()
        print("            loss_class:", loss_class_c, "   loss_cover:", loss_cover_c)

        if step % 5000 == 0:
            print("save weight at step:%d" % (step))
            util.save_weight(class_net, step, ckpt_class_path)
            print("save weight at step:%d" % (step))
            util.save_weight(cover_net, step, ckpt_cover_path)
            print("save optimizer weight at step:%d" % (step))
            util.save_weight(optimizer, step, ckpt_optimizer_path)


def view():
    class_net = ResNet18()
    class_net.cuda().eval()

    cover_net = COVER_NET()
    cover_net.cuda().eval()

    ckpt_cover_path = "./all_ckpts/ckpt_cover/"
    data_path, step = util.load_weight(ckpt_cover_path)
    if step:
        cover_net.load_state_dict(torch.load(data_path))

    ckpt_class_path = "./all_ckpts/ckpt_class/"
    data_path, step = util.load_weight(ckpt_class_path)
    if step:
        class_net.load_state_dict(torch.load(data_path))

    batch = 32
    criterion = nn.CrossEntropyLoss()

    # data_path = '/media/gdh-95/data/CT_2d_note_slice/train/'
    data_path = '/home/cuiyang/data/CT_2d_lidc/train/'
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.ImageFolder(data_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=1)

    dataiter = iter(train_loader)
    lenth = len(train_set)
    print('img_num', lenth)
    out_path = './cover_2d_out/'

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    save_num = 0
    error_num = 0
    break_flag = 0
    area_add = 0
    batch_num = 0
    for images0, labels in dataiter:
        # print('DEBUG images0 shape: ',images0.shape)
        # print('labels: ',labels)
        if break_flag:
            break
        batch_num += 1
        images0 = images0.cuda()
        labels0 = labels.cpu().data.numpy()

        images = images0 * 1
        labels = labels.float().cuda()
        cover_mask = cover_net(images)

        class_out = class_net(images, cover_mask)
        class_out = torch.mean(class_out, dim=1)
        labels = labels + 1
        # print("labels:", labels)
        # print("class_out:", class_out)
        # print(torch.min(class_out))

        loss_class = (class_out - labels)
        loss_class = (loss_class * loss_class).mean()
        loss_cover = torch.mean(cover_mask)

        area_add += loss_cover.cpu().data.numpy()

        error_b = (class_out - labels).abs().sum().cpu().data.numpy()
        error_num += error_b
        loss_class_c = loss_class.cpu().data.numpy()
        loss_cover_c = loss_cover.cpu().data.numpy()

        print("loss_class:", loss_class_c, "  loss_cover:", loss_cover_c)
        batch_max = images.size(0)
        for b in range(batch_max):
            image_one = images0[b:b + 1, :, :, :]
            cover_one = cover_mask[b:b + 1, :, :, :]
            overlay_one = image_one * 1
            overlay_one[:, 1, :, :] = overlay_one[:, 1, :, :] * (1 - cover_one)
            image_one = util.torch2numpy(image_one * 255)
            cover_one = np.squeeze(util.torch2numpy(cover_one * 255))
            overlay_one = util.torch2numpy(overlay_one * 255)
            # print labels0[b]
            # print str(labels0[b])
            save_path = out_path + str(labels0[b]) + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # cv2.imwrite(save_path + str(save_num) + '_a.jpg', image_one)
            # cv2.imwrite(save_path + str(save_num) + '_b.jpg', overlay_one)
            # cv2.imwrite(save_path + str(save_num) + '_c.jpg', cover_one)
            save_num += 1
    print("correct:", 1-float(error_num) / float(save_num))
    print("error:", float(error_num) / float(save_num))
    print("area:", area_add / float(batch_num))


def view_for_pcl(data_url, save_url, all_ckpt_path):
    class_net = ResNet18()
    class_net.cuda().eval()

    cover_net = COVER_NET()
    cover_net.cuda().eval()

    ckpt_cover_path = os.path.join(all_ckpt_path, "ckpt_cover")
    data_path, step = util.load_weight(ckpt_cover_path)
    if step:
        cover_net.load_state_dict(torch.load(data_path))

    ckpt_class_path = os.path.join(all_ckpt_path, "ckpt_class")
    data_path, step = util.load_weight(ckpt_class_path)
    if step:
        class_net.load_state_dict(torch.load(data_path))

    batch = 32
    criterion = nn.CrossEntropyLoss()

    data_path = data_url
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.ImageFolder(data_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=1)

    dataiter = iter(train_loader)
    lenth = len(train_set)
    print('img_num', lenth)
    out_path = save_url

    if os.path.exists(out_path):
        flush_dir(out_path)
    else:
        os.mkdir(out_path)

    save_num = 0
    error_num = 0
    break_flag = 0
    area_add = 0
    batch_num = 0
    for images0, labels in dataiter:
        # print('DEBUG images0 shape: ',images0.shape)
        # print('labels: ',labels)
        if break_flag:
            break
        batch_num += 1
        images0 = images0.cuda()
        labels0 = labels.cpu().data.numpy()

        images = images0 * 1
        labels = labels.float().cuda()
        cover_mask = cover_net(images)

        class_out = class_net(images, cover_mask)
        class_out = torch.mean(class_out, dim=1)
        labels = labels + 1
        # print("labels:", labels)
        # print("class_out:", class_out)
        # print(torch.min(class_out))

        loss_class = (class_out - labels)
        loss_class = (loss_class * loss_class).mean()
        loss_cover = torch.mean(cover_mask)

        area_add += loss_cover.cpu().data.numpy()

        error_b = (class_out - labels).abs().sum().cpu().data.numpy()
        error_num += error_b
        loss_class_c = loss_class.cpu().data.numpy()
        loss_cover_c = loss_cover.cpu().data.numpy()

        print("loss_class:", loss_class_c, "  loss_cover:", loss_cover_c)
        batch_max = images.size(0)
        for b in range(batch_max):
            image_one = images0[b:b + 1, :, :, :]
            cover_one = cover_mask[b:b + 1, :, :, :]
            overlay_one = image_one * 1
            overlay_one[:, 1, :, :] = overlay_one[:, 1, :, :] * (1 - cover_one)
            image_one = util.torch2numpy(image_one * 255)
            cover_one = np.squeeze(util.torch2numpy(cover_one * 255))
            overlay_one = util.torch2numpy(overlay_one * 255)
            # print labels0[b]
            # print str(labels0[b])
            save_path = out_path + str(labels0[b]) + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image_pil = Image.fromarray(np.uint8(image_one))
            image_pil.save(save_path + str(save_num) + '_a.jpg')
            overlay_pil = Image.fromarray(np.uint8(overlay_one))
            overlay_pil.save(save_path + str(save_num) + '_b.jpg')
            cover_pil = Image.fromarray(np.uint8(cover_one))
            cover_pil.save(save_path + str(save_num) + '_c.jpg')
            # cv2.imwrite(save_path + str(save_num) + '_a.jpg', image_one)
            # cv2.imwrite(save_path + str(save_num) + '_b.jpg', overlay_one)
            # cv2.imwrite(save_path + str(save_num) + '_c.jpg', cover_one)
            # Image.fromarray(image_one).save(save_path + str(save_num) + '_a.jpg')
            # Image.fromarray(overlay_one).save(save_path + str(save_num) + '_b.jpg')
            # Image.fromarray(cover_one).save(save_path + str(save_num) + '_c.jpg')
            save_num += 1
    print("correct:", 1-float(error_num) / float(save_num))
    print("error:", float(error_num) / float(save_num))
    print("area:", area_add / float(batch_num))




# class_net = None
# cover_net = None

# def init_tracer_model():
#     global class_net
#     global cover_net
#     class_net = ResNet18()
#     class_net.cuda().eval()

#     cover_net = COVER_NET()
#     cover_net.cuda().eval()

#     ckpt_cover_path = config_tracer['cover_model']
#     data_path, step = util.load_weight(ckpt_cover_path)
#     if step:
#         cover_net.load_state_dict(torch.load(data_path))

#     ckpt_class_path = config_tracer['class_model']
#     data_path, step = util.load_weight(ckpt_class_path)
#     if step:
#         class_net.load_state_dict(torch.load(data_path))
#     pass



def view_new(patientname,savename):
    class_net = ResNet18()
    class_net.cuda().eval()

    cover_net = COVER_NET()
    cover_net.cuda().eval()

    ckpt_cover_path = config_tracer['cover_model']
    data_path, step = util.load_weight(ckpt_cover_path)
    if step:
        cover_net.load_state_dict(torch.load(data_path))

    ckpt_class_path = config_tracer['class_model']
    data_path, step = util.load_weight(ckpt_class_path)
    if step:
        class_net.load_state_dict(torch.load(data_path))

    # if cover_net is None:
    #     print("cover_net is None.")
    #     return 
        

    batch = 64
    criterion = nn.CrossEntropyLoss()

    # data_path = '/media/gdh-95/data/CT_2d_note_slice/train/'
    # data_path = os.path.join(config_tracer['feat_save_path'], patientname)
    data_path = os.path.join(config_tracer['extractfiles'], patientname) # for hw docker
    transform = transforms.Compose([transforms.ToTensor()])
    # train_set = datasets.ImageFolder(data_path, transform=transform)
    test_set = MyDataset(data_path, transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    lenth = len(test_set)
    print('img_num', lenth)
    out_path = os.path.join(config_tracer['output_path'], patientname)

    if os.path.exists(out_path):
        flush_dir(out_path)
    else:
        os.mkdir(out_path)

    save_num = 0

    for i,(filename, image_tensor, label) in enumerate(test_loader):
        image_tensor = image_tensor.cuda()
        # image_tensor = torch.unsqueeze(image_tensor, 0)
        # label = label.cpu().data.numpy()
        image = image_tensor * 1
        # label = label.float().cuda()
        cover_mask = cover_net(image)

        overlay = image * 1
        overlay[:,1, :, :] = overlay[:,1, :, :] * (1 - cover_mask)
        image_one = util.torch2numpy(image * 255)
        cover_one = np.squeeze(util.torch2numpy(cover_mask * 255))
        overlay_one = util.torch2numpy(overlay * 255)
        # print labels0[b]
        if not savename:
            savename=patientname
        # cv2.imencode('.png', image_one)[1].tofile(
        #     os.path.join(out_path, str(i) + '_origin.png'))
        # cv2.imencode('.png', overlay_one)[1].tofile(
        #     os.path.join(out_path, str(i) + '_overlay.png'))
        # cv2.imencode('.png', cover_one)[1].tofile(
        #     os.path.join(out_path, str(i) + '_cover.png'))

        # cv2.imwrite(os.path.join(out_path, savename + '_origin.jpg'), image_one)
        # cv2.imwrite(os.path.join(out_path, savename + '_overlay.jpg'), overlay_one)
        # cv2.imwrite(os.path.join(out_path, savename + '_cover.jpg'), cover_one)
        save_num += 1

    print('save num', save_num)
    print('view_new OK')
    return save_num

def view_single(filename, savename):
    class_net = ResNet18()
    class_net.cuda().eval()

    cover_net = COVER_NET()
    cover_net.cuda().eval()

    ckpt_cover_path = config_tracer['cover_model']
    data_path, step = util.load_weight(ckpt_cover_path)
    if step:
        cover_net.load_state_dict(torch.load(data_path))

    ckpt_class_path = config_tracer['class_model']
    data_path, step = util.load_weight(ckpt_class_path)
    if step:
        class_net.load_state_dict(torch.load(data_path))

    input_path = config_tracer['input_path']
    out_path = config_tracer['output_path']
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    img_path = os.path.join(input_path, filename)
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(default_loader(img_path))

    print('DEBUG: SHAPE ', img_tensor.shape)

    image_tensor = img_tensor.cuda()
    # image_tensor = torch.unsqueeze(image_tensor, 0)

    image = image_tensor * 1
    cover_mask = cover_net(image)

    overlay = image * 1
    overlay[:, 1, :, :] = overlay[:, 1, :, :] * (1 - cover_mask)

    image_one = util.torch2numpy(image * 255)
    cover_one = np.squeeze(util.torch2numpy(cover_mask * 255))
    overlay_one = util.torch2numpy(overlay * 255)

    if not savename:
        savename = 'unkown'
    # cv2.imwrite(os.path.join(out_path, savename+'_origin.jpg', image_one))
    # cv2.imwrite(os.path.join(out_path, savename+'_overlay.jpg', overlay_one))
    # cv2.imwrite(os.path.join(out_path, savename+'_cover.jpg', cover_one))

    print('view single ok')




def flush_dir(dirpath):
    for f in os.listdir(dirpath):
        f_path = os.path.join(dirpath, f)
        shutil.rmtree(f_path)
    print("flush", dirpath, "complete")
