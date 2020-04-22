import os
import sys
import torch
import torch.nn.functional as F
from eval import eval_model

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir,save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix,steps)
    torch.save(model.state_dict(),save_path)


def train_model(train_iter, dev_iter, model, args):

    if args.cuda:
        model.cuda(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()

    print('training...')
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            feature, target = batch.text, batch.label  # (W,N) (N)
            with torch.no_grad():
                feature = feature.t()

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                result = torch.max(logit, 1)[1].view(target.size())
                corrects = (result.data == target.data).sum()
                accuracy = corrects * 100.0 / batch.batch_size
                sys.stdout.write('\r Epoch[{}] - Batch[{}] - loss: {:.6f} acc: {:.4f}$({}/{})'.format(epoch,
                                                                                                      steps,
                                                                                         loss.data.item(),
                                                                                         accuracy,
                                                                                         corrects,
                                                                                         batch.batch_size))
            if steps % args.dev_interval == 0:
                dev_acc = eval_model(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)