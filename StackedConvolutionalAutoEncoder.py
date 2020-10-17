# _*_ coding: utf-8 _*_
"""
    Author: Kwong
    Create time: 2020/10/10 13:00
"""

import os
import argparse
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from collections import OrderedDict

model_save = "./Model_save"
pic_save = "./Pic_save/StackedConvolutionalAutoEncoder/"
if not os.path.exists(model_save):
    os.mkdir(model_save)
if not os.path.exists("./Pic_save"):
    os.mkdir("./Pic_save")
if not os.path.exists(pic_save):
    os.mkdir(pic_save)


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(32, 32, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(32, 32, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(32, 32, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(32, 32, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(32, 32, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(32 * 2, 32, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(1, 32, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(32, 32, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(32, 32, [9, 1])),
        ]))
        self.encoder2 = ConvBN(1, 32, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(32 * 2, 1, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.fc = nn.Linear(784, 32)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = out.view(-1, 784)
        out = self.fc(out)
        out = self.sig(out)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(32, 784)
        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(1, 32, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock()),
            ("CRBlock2", CRBlock()),
        ])
        self.decoder_feature = nn.Sequential(decoder)
        self.out_cov = ConvBN(32, 1, 3)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = x.view(-1, 32)
        out = self.fc(out)
        out = self.sig(out)
        out = out.view(-1, 1, 28, 28)
        out = self.decoder_feature(out)
        out = self.out_cov(out)
        out = self.sig(out)
        return out


class StackedConvolutionalAE(nn.Module):
    def __init__(self, ):
        super(StackedConvolutionalAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out


def train(args):
    # setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(17)

    # load data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    data = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # create model
    model = StackedConvolutionalAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # start train
    best_loss = 1e10
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for idx, (img, _) in enumerate(train_loader):
            img = img.to(device)
            optimizer.zero_grad()

            prediction = model(img)
            loss = criterion(prediction.view(img.shape[0], -1), img.view(img.shape[0], -1))
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            # ---------- print log ---------------
            if idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.6f}'.format(
                    epoch, idx, len(train_loader), 100. * idx / len(train_loader), loss.item()))
            del loss

        train_loss /= len(train_loader)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))

        # ------------ save model ---------------
        if best_loss > train_loss:
            best_loss = train_loss
            try:
                torch.save(model.module.state_dict(), os.path.join(model_save, "StackedConvolutionalAutoEncoder.pkl"))
            except:
                torch.save(model.state_dict(), os.path.join(model_save, "StackedConvolutionalAutoEncoder.pkl"))
            print("Save model!")

        # ----------- save image ---------------
        if epoch % 5 == 0:
            img_save(img, os.path.join(pic_save, "img_%d.png" % (epoch)))
            img_save(prediction, os.path.join(pic_save, "pred_%d.png" % (epoch)))


def img_save(image, url):
    image = image.view(image.shape[0], 1, 28, 28).cpu().data
    save_image(image, url, nrow=16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stacked Convolutional AE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status (default: 50)')
    args = parser.parse_args()
    train(args)
