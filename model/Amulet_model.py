
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision


class RFC(nn.Module):

    def __init__(self, resolution=0):
        super().__init__()
        if resolution == 0:
            raise ValueError('resolution should be in [16, 32, 64, 128, 256]')
        elif resolution == 16:
            self.layer1 = nn.Conv2d(64, 64, 16, stride=16)
            self.layer2 = nn.Conv2d(128, 64, 8, stride=8)
            self.layer3 = nn.Conv2d(256, 64, 4, stride=4)
            self.layer4 = nn.Conv2d(512, 64, 2, stride=2)
            self.layer5 = nn.Conv2d(512, 64, 1)
            self.upsample = nn.ConvTranspose2d(64, 64, 16, stride=16)

        elif resolution == 32:
            self.layer1 = nn.Conv2d(64, 64, 8, stride=8)
            self.layer2 = nn.Conv2d(128, 64, 4, stride=4)
            self.layer3 = nn.Conv2d(256, 64, 2, stride=2)
            self.layer4 = nn.Conv2d(512, 64, 1)
            self.layer5 = nn.ConvTranspose2d(512, 64, 2, stride=2)
            self.upsample = nn.ConvTranspose2d(64, 64, 8, stride=8)

        elif resolution == 64:
            self.layer1 = nn.Conv2d(64, 64, 4, stride=4)
            self.layer2 = nn.Conv2d(128, 64, 2, stride=2)
            self.layer3 = nn.Conv2d(256, 64, 1)
            self.layer4 = nn.ConvTranspose2d(512, 64, 2, stride=2)
            self.layer5 = nn.ConvTranspose2d(512, 64, 4, stride=4)
            self.upsample = nn.ConvTranspose2d(64, 64, 4, stride=4)

        elif resolution == 128:
            self.layer1 = nn.Conv2d(64, 64, 2, stride=2)
            self.layer2 = nn.Conv2d(128, 64, 1)
            self.layer3 = nn.ConvTranspose2d(256, 64, 2, stride=2)
            self.layer4 = nn.ConvTranspose2d(512, 64, 4, stride=4)
            self.layer5 = nn.ConvTranspose2d(512, 64, 8, stride=8)
            self.upsample = nn.ConvTranspose2d(64, 64, 2, stride=2)

        elif resolution == 256:
            self.layer1 = nn.Conv2d(64, 64, 1)
            self.layer2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.layer3 = nn.ConvTranspose2d(256, 64, 4, stride=4)
            self.layer4 = nn.ConvTranspose2d(512, 64, 8, stride=8)
            self.layer5 = nn.ConvTranspose2d(512, 64, 16, stride=16)
            self.upsample = nn.ConvTranspose2d(64, 64, 1, stride=1)

        else:
            raise ValueError('resolution should be in [16, 32, 64, 128, 256]')
        self.conv_out = nn.Conv2d(320, 64, 1)
        self.bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, feature1, feature2, feature3, feature4, feature5):
        feature1 = self.layer1(feature1)
        feature2 = self.layer2(feature2)
        feature3 = self.layer3(feature3)
        feature4 = self.layer4(feature4)
        feature5 = self.layer5(feature5)
        feature = torch.cat([feature1, feature2, feature3, feature4, feature5], 1)
        feature = self.conv_out(feature)
        feature = self.bn(feature)
        feature = self.relu(feature)
        feature = self.upsample(feature)
        # Upsample(deconvolution) layer was placed after RFC block in original paper, but for convenience
        # I just put it here.
        return feature


class Amulet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(  # VGG16 net conv block1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(  # VGG16 net conv block2
            nn.MaxPool2d(2, stride=2),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(  # VGG16 net conv block3
            nn.MaxPool2d(2, stride=2),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(  # VGG16 net conv block4
            nn.MaxPool2d(2, stride=2),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(  # VGG16 net conv block5
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.rfc1 = RFC(resolution=256)
        self.rfc2 = RFC(resolution=128)
        self.rfc3 = RFC(resolution=64)
        self.rfc4 = RFC(resolution=32)
        self.rfc5 = RFC(resolution=16)

        self.catconv1 = nn.Conv2d(64, 2, 3, padding=1)
        self.catconv2 = nn.Conv2d(66, 2, 3, padding=1)
        self.catconv3 = nn.Conv2d(66, 2, 3, padding=1)
        self.catconv4 = nn.Conv2d(66, 2, 3, padding=1)
        self.catdeconv = nn.ConvTranspose2d(66, 2, 3, padding=1)

        self.out_conv = nn.Sequential(
            nn.Conv2d(10, 64, 1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1)
        )
        self.vgg16_weight_init()
        # print('model initialization complete')

    def forward(self, imgs):
        feature1 = self.conv1(imgs)
        feature2 = self.conv2(feature1)
        feature3 = self.conv3(feature2)
        feature4 = self.conv4(feature3)
        feature5 = self.conv5(feature4)
        rfc1 = self.rfc1(feature1, feature2, feature3, feature4, feature5)
        rfc2 = self.rfc2(feature1, feature2, feature3, feature4, feature5)
        rfc3 = self.rfc3(feature1, feature2, feature3, feature4, feature5)
        rfc4 = self.rfc4(feature1, feature2, feature3, feature4, feature5)
        rfc5 = self.rfc5(feature1, feature2, feature3, feature4, feature5)
        rfc5 = self.catconv1(rfc5)
        output1 = rfc5
        cat1 = torch.cat([rfc5, rfc4], 1)
        cat1 = self.catconv2(cat1)
        output2 = cat1
        cat2 = torch.cat([cat1, rfc3], 1)
        cat2 = self.catconv3(cat2)
        output3 = cat2
        cat3 = torch.cat([cat2, rfc2], 1)
        cat3 = self.catconv4(cat3)
        output4 = cat3
        cat4 = torch.cat([cat3, rfc1], 1)
        cat4 = self.catdeconv(cat4)
        output5 = cat4
        cat = torch.cat([output1, output2, output3, output4, output5], 1)
        output = self.out_conv(cat)
        return output, output1, output2, output3, output4, output5

    def vgg16_weight_init(self):
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        features = list(self.conv1.children())
        features.extend(list(self.conv2.children()))
        features.extend(list(self.conv3.children()))
        features.extend(list(self.conv4.children()))
        features.extend(list(self.conv5.children()))
        features = nn.Sequential(*features)

        for layer_1, layer_2 in zip(vgg16.features, features):
            if (isinstance(layer_1, nn.Conv2d) and
                    isinstance(layer_2, nn.Conv2d)):
                assert layer_1.weight.size() == layer_2.weight.size()
                assert layer_1.bias.size() == layer_2.bias.size()
                layer_2.weight.data = layer_1.weight.data
                layer_2.bias.data = layer_1.bias.data


if __name__ == '__main__':
    testnet = Amulet()
    img = PIL.Image.open('test.jpg', 'r')
    img = img.resize((256, 256))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 1, 2))
    img = torch.from_numpy(img).float()
    outputs = testnet.forward(img)
