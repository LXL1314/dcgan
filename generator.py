from mxnet.gluon import nn

class Generator(nn.Block):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Conv2DTranspose(1024, kernel_size=4, strides=1, padding=0),
                     nn.BatchNorm(),
                     nn.Activation('relu'),

                     nn.Conv2DTranspose(512, kernel_size=4, strides=2, padding=1),
                     nn.BatchNorm(),
                     nn.Activation('relu'),

                     nn.Conv2DTranspose(256, kernel_size=4, strides=2, padding=1),
                     nn.BatchNorm(),
                     nn.Activation('relu'),

                     nn.Conv2DTranspose(128, kernel_size=4, strides=2, padding=1),
                     nn.BatchNorm(),
                     nn.Activation('relu'),

                     nn.Conv2DTranspose(3, kernel_size=4, strides=2, padding=1),
                     nn.Activation('tanh'))

        def forward(self, input_z):  # input_z:(batch_size, 100, 1, 1)
            return self.net(input_z)