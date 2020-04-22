import torch
import torch.nn.functional as F

def eval_model(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        with torch.no_grad():
            feature = feature.t()

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target)

        avg_loss += loss.data
        result = torch.max(logit, 1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f} acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size))

    return accuracy