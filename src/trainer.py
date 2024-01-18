import time
from utils import *
from runx.logx import logx
from os.path import join as ospj


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        batch_img, batch_label = batch
        batch_img = batch_img.to(device)
        # batch_text = batch_text.permute(1, 0)
        batch_label = batch_label.to(device)

        optimizer.zero_grad()

        outputs = model(batch_img)
        _, pred_label = torch.max(outputs, 1)

        loss = criterion(outputs, batch_label)

        acc = multilabel_acc(pred_label.cpu().numpy(), batch_label.cpu().numpy())

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        # print(acc.item())

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    epoch_label = None
    epoch_pred = None

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            batch_img, batch_label = batch
            batch_img = batch_img.to(device)
            # batch_text = batch_text.permute(1, 0)
            batch_label = batch_label.to(device)

            outputs = model(batch_img)
            _, pred_label = torch.max(outputs, 1)

            loss = criterion(outputs, batch_label)

            if epoch_label is None:
                epoch_label = batch_label
                epoch_pred = pred_label
            else:
                epoch_label = torch.cat((epoch_label, batch_label), 0)
                epoch_pred = torch.cat((epoch_pred, pred_label), 0)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), multilabel_metric(pred_label.cpu().numpy(), batch_label.cpu().numpy())


def train_model(config, model, optimizer, criterion, train_iterator, valid_iterator, test_iterator):
    model = model.to(config.device)
    criterion = criterion.to(config.device)

    best_valid_loss = float('inf')
    N_EPOCHS = config.epochs
    best_model_path = ospj(config.ckpt_dir, 'best_model.pt')
    last_model_path = ospj(config.ckpt_dir, 'last_model.pt')
    localtime = time.asctime(time.localtime(time.time()))
    logx.msg('======================Start Train Model [{}]======================'.format(localtime))

    train_acc_cache = {'Value': []}
    valid_acc_cache = {'Value': []}

    # for epoch in range(N_EPOCHS):
    #
    #     start_time = time.time()
    #     train_loss, train_acc = train(model, train_iterator, optimizer, criterion, config.device)
    #     # logx.add_scalar('train_loss', train_loss, epoch)
    #     # logx.add_scalar('train_acc', train_acc, epoch)
    #     train_acc_cache['Value'].append(train_acc)
    #     valid_loss, valid_mm = evaluate(model, valid_iterator, criterion, config.device)
    #     # logx.add_scalar('valid_loss', valid_loss, epoch)
    #     valid_acc_cache['Value'].append(valid_mm['acc'])
    #     end_time = time.time()
    #
    #     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #
    #     if valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         torch.save(model.state_dict(), best_model_path)
    #     if epoch == N_EPOCHS - 1:
    #         torch.save(model.state_dict(), last_model_path)
    #
    #     logx.msg('Epoch: {} | Epoch Time: {}m {}s'.format(epoch + 1, epoch_mins, epoch_secs))
    #     logx.msg('Train Loss: {} | Train Acc: {}%'.format(train_loss, train_acc * 100))
    #     logx.msg('Val. Loss: {} | Val. Acc: {}%'.format(valid_loss, valid_mm['acc'] * 100))
    #
    # save_csv(train_acc_cache, ospj(config.log_dir, 'train_acc.csv'))
    # save_csv(valid_acc_cache, ospj(config.log_dir, 'valid_acc.csv'))
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_mm = evaluate(model, test_iterator, criterion, config.device)

    # print('Test Loss: {:.3f} | Test Acc: {:.2f}%'.format(test_loss, test_acc * 100))
    logx.msg('Test Loss: {:.3f} | Test Acc: {:.2f}%'.format(test_loss, test_mm['acc'] * 100))
    print(test_mm)
    save_json(test_mm, ospj(config.log_dir, 'test_mm.json'))

    localtime = time.asctime(time.localtime(time.time()))
    logx.msg('======================Finish Train Model [{}]======================'.format(localtime))
    return model
