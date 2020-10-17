# _*_ coding: utf-8 _*_
"""
    Author: Kwong
    Create time: 2020/10/13 8:53 
"""

import os
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


model_save = "./Model_save"
pic_save = "./Pic_save/DenoisingAutoEncoder/"
if not os.path.exists(model_save):
    os.mkdir(model_save)
if not os.path.exists("./Pic_save"):
    os.mkdir("./Pic_save")
if not os.path.exists(pic_save):
    os.mkdir(pic_save)


class DenoisingAE(nn.Module):
    def __init__(self):
        super(DenoisingAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 300),
            nn.ReLU(True),
            nn.Linear(300, 32),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 300),
            nn.ReLU(True),
            nn.Linear(300, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode


def add_noise(img):
    noise = torch.randn(img.shape) * 0.3
    noise_img = img + noise
    noise_img[noise_img > 1] = 1
    noise_img[noise_img < 0] = 0
    return noise_img


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
    model = DenoisingAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # start train
    best_loss = 1e10
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for idx, (img, _) in enumerate(train_loader):
            img = img.view(-1, 784)
            noise_img = add_noise(img).to(device).float()
            img = img.to(device)
            optimizer.zero_grad()

            prediction = model(noise_img)
            loss = criterion(prediction, img)
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
                torch.save(model.module.state_dict(), os.path.join(model_save, "DenoisingAutoEncoder.pkl"))
            except:
                torch.save(model.state_dict(), os.path.join(model_save, "DenoisingAutoEncoder.pkl"))
            print("Save model!")

        # ----------- save image ---------------
        if epoch % 5 == 0:
            img_save(img, os.path.join(pic_save, "img_%d.png" % (epoch)))
            img_save(prediction, os.path.join(pic_save, "pred_%d.png" % (epoch)))
            img_save(noise_img, os.path.join(pic_save, "noise_img_%d.png" % (epoch)))


def img_save(image, url):
    image = image.view(image.shape[0], 1, 28, 28).cpu().data
    save_image(image, url, nrow=16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Denoising AE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--expect_tho', type=float, default=0.05,
                        help='The sparse degree (default: 5e-2)')
    parser.add_argument('--beta', type=float, default=3,
                        help='Consideration degree of sparse term (default: 3)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status (default: 50)')
    args = parser.parse_args()
    train(args)

