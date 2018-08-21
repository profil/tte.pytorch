import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
from torch import optim
from torch.autograd import Variable

from ignite.trainer import Trainer
from ignite.engine import Events
from ignite.evaluator import Evaluator
from ignite.handlers.evaluate import Evaluate
from ignite.handlers.logging import log_simple_moving_average

from tte.model import TTE
from tte.weibull import weibull_loglikelihood

from .progress import Progress
from .cmapss import CMAPSSData
from .utils import collate_fn

USE_CUDA = torch.cuda.is_available()
print(f"Using cuda: {USE_CUDA}")

def run():
    train_dataloader = data.DataLoader(CMAPSSData(train=True), shuffle=True,
                                       batch_size=300, pin_memory=USE_CUDA,
                                       collate_fn=collate_fn)

    validation_dataloader = data.DataLoader(CMAPSSData(train=False), shuffle=True,
                                            batch_size=1000, pin_memory=USE_CUDA,
                                            collate_fn=collate_fn)

    model = TTE(24, 128, n_layers=1, dropout=0.2)
    print(model)
    if USE_CUDA:
        model.cuda()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    loss_fn = weibull_loglikelihood

    def training_update(batch):
        model.train()
        optimizer.zero_grad()
        inputs, lengths, targets = batch
        if USE_CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs, lengths)
        loss = loss_fn(outputs, targets, lengths)
        loss.backward()
        optimizer.step()
        return loss.data[0]

    def validation_inference(batch):
        model.eval()
        inputs, lengths, targets = batch
        if USE_CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        outputs = model(inputs, lengths)
        loss = loss_fn(outputs, targets, lengths)
        return loss.data[0], outputs.data[:, :, 0], outputs.data[:, :, 1], targets.data

    trainer = Trainer(training_update)
    evaluator = Evaluator(validation_inference)
    progress = Progress()
    plot_interval = 1

    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              Evaluate(evaluator, validation_dataloader, epoch_interval=1))

    @trainer.on(Events.EPOCH_STARTED)
    def epoch_started(trainer):
        print('Epoch {:4}/{}'.format(trainer.current_epoch,
                                   trainer.max_epochs), end='')

    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_completed(trainer):
        if trainer.current_iteration % plot_interval == 0:
            avg_loss = trainer.history.simple_moving_average(window_size=100)
            values = [('iter', trainer.current_iteration),
                      ('loss', avg_loss)]
            progress.update(values)

    @evaluator.on(Events.COMPLETED)
    def epoch_completed(evaluator):
        history = evaluator.history[0]
        loss = history[0]
        alpha = history[1]
        beta = history[2]
        target = history[3]

        mae = torch.mean(torch.abs(alpha - target[:, :, 0]))
        alpha = alpha.mean()
        beta = beta.mean()

        values = [('val_loss', loss),
                  ('mae', mae),
                  ('alpha', alpha),
                  ('beta', beta)]
        progress.update(values, end=True)

    trainer.run(train_dataloader, max_epochs=600)

    return model


def weibull_pdf(alpha, beta, t):
    return (beta/alpha) * (t/alpha)**(beta-1)*np.exp(- (t/alpha)**beta)

def weibull_median(alpha, beta):
    return alpha*(-np.log(.5))**(1/beta)

def weibull_mean(alpha, beta):
    return alpha * math.gamma(1 + 1/beta)

def weibull_mode(alpha, beta):
    if np.all(beta > 1):
        return alpha * ((beta - 1) / beta)**(1 / beta)
    else:
        return 0

def plot(model, inputs):
    lengths = [inputs.size(0)]
    output = model(Variable(inputs.cuda().unsqueeze(1)), lengths).data.cpu().squeeze(1).numpy()

    t = np.arange(250)
    for i, o in enumerate(output):
        alpha = o[0]
        beta = o[1]
        mode = weibull_mode(alpha, beta)
        y_max = weibull_pdf(alpha, beta, mode)

        plt.plot(t, weibull_pdf(alpha, beta, t), label=i)
        #plt.scatter(T, weibull_pdf(alpha,beta, T), s=100)
        #plt.vlines(mode, ymin=0, ymax=y_max, linestyles='--')

    plt.legend(loc='best')
    plt.show()

    return output

if __name__ == "__main__":
    run()
