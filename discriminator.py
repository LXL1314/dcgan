from mxnet.gluon import nn

class Discriminator(nn.Block):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Conv2D(128, kernel_size=4, strides=2, padding=1),
                     nn.LeakyReLU(),

                     nn.Conv2D(256, kernel_size=4, strides=2, padding=1),
                     nn.BatchNorm(),
                     nn.LeakyReLU(),

                     nn.Conv2D(512, kernel_size=4, strides=2, padding=1),
                     nn.BatchNorm(),
                     nn.LeakyReLU(),

                     nn.Conv2D(1024, kernel_size=4, strides=2, padding=1),
                     nn.BatchNorm(),
                     nn.LeakyReLU(),

                     nn.Conv2D(1, kernel_size=4, strides=1, padding=0),
                     nn.BatchNorm(),
                     nn.LeakyReLU(),

                     nn.Activation('sigmoid'))

        # 得到的结果是（batch_size, 1, 1, 1）

    def forward(self, input):  # input:(batch_size, 3, 64, 64)
        out = self.net(input)  #out:（batch_size, 1, 1, 1）
        return out.reshape((out.shape[0], ))  # 返回的结果：（batch_size, ), 一个一维数组
