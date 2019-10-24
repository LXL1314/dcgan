from mxnet.gluon.data.vision import transforms, datasets
from mxnet.gluon.data import DataLoader
from mxnet.gluon import loss as gloss, Trainer
from mxnet import nd, init, autograd
import generator, discriminator

batch_size = 128
lr = 0.0002
momentum = 0.5
epoches = 5

dataset = datasets.ImageFolderDataset(root='data/celeba',
                                      transform=transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                                                         (0.5, 0.5, 0.5))]))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=4)

netG = generator.Generator()
netD = discriminator.Discriminator()

netG.initialize(init=init.Normal(sigma=0.02))
netD.initialize(init=init.Normal(sigma=0.02))

trainerG = Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'momentum': momentum})
trainerD = Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'momentum': momentum})

loss = gloss.SigmoidBCELoss()

for epoch in range(epoches):
    for i, batch in enumerate(data_loader):
        data_real = batch[0]
        labels_real = nd.ones(shape=(batch_size, ))

        z = nd.random.randn(batch_size, 100, 1, 1)
        data_fake = netG(z)
        labels_fake = nd.zeros(shape=(batch_size, ))
        with autograd.record():
            preds_real = netD(data_real)
            lo_real = loss(preds_real, labels_real)
            preds_fake = netD(data_fake)
            lo_fake = loss(preds_fake, labels_fake)
            err_D = preds_real + preds_fake
        err_D.backward()
        trainerD.step(batch_size)
        D_x = preds_real.mean().asscalar()
        D_G_z_before = preds_fake.mean().asscalar()

        with autograd.record():
            preds_fake = netD(data_fake)
            err_G = loss(preds_fake, labels_real)
        err_G.backward()
        trainerG.step(batch_size)
        D_G_z_after = preds_fake.mean().asscalar()

        if (i + 1) % 50 == 0:
            print("[%d/%d] [%d/%d]\t errD: %.4f\t errG: %.4f\tD(x): %.4f\tD(G(z)) before: %.4f\tafter: %.4f"
                  % (epoch, epoches, i, len(data_loader), err_D.asscalar(), err_G.asscalar(),
                     D_x, D_G_z_before, D_G_z_after))