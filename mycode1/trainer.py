import math
import time
import copy
from utils import *
from metrics import *
from adj_dis_matrix import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def record_loss(loss_file, loss):
    with open(loss_file, 'a') as f:
        line = "{:.4f}\n".format(loss)
        f.write(line)


class Trainer(object):
    def __init__(self,
                 args,
                 generator, discriminator, discriminator_rf, loss_D,
                 optimizer_G, optimizer_D, optimizer_D_RF,
                 lr_scheduler_G, lr_scheduler_D, lr_scheduler_D_RF,norm_dis_matrix):

        super(Trainer, self).__init__()
        self.args = args
        self.num_nodes = args.num_nodes
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_rf = discriminator_rf
        self.loss_D = loss_D
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.optimizer_D_RF = optimizer_D_RF
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        self.lr_scheduler_D_RF = lr_scheduler_D_RF
        self.norm_dis_matrix = norm_dis_matrix
        self.best_path = os.path.join(self.args.log_dir, 'best_mod1el.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')  # when plot=True

        # log info
        if os.path.isdir(args.log_dir) == False:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info(f"Argument: {args}")
        for arg, value in sorted(vars(args).items()):
            self.logger.info(f"{arg}: {value}")

    def train(self):

        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
         testY, mean, std) = load_data(self.args)
        num_train, _, _ = trainX.shape
        num_val = valX.shape[0]

        train_num_batch = math.ceil(num_train / self.args.batch_size)
        val_num_batch = math.ceil(num_val / self.args.batch_size)

        # loss file
        loss_file = '{}_{}_val_loss.txt'.format(self.args.model, self.args.ds)
        if os.path.exists(loss_file):
            os.remove(loss_file)
            print('Recreate {}'.format(loss_file))

        start_time = time.time()
        for epoch in range(1, self.args.max_epoch + 1):
            total_loss_G = 0
            total_loss_D = 0


            permutation = torch.randperm(num_train)
            trainX = trainX[permutation]
            trainTE = trainTE[permutation]
            trainY = trainY[permutation]
            self.generator.train()
            self.discriminator.train()
            for batch_idx in range(train_num_batch):
                # 如果是最后一批，不处理
                if batch_idx == train_num_batch - 1 and num_train % self.args.batch_size != 0:
                    break
                start_idx = batch_idx * self.args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * self.args.batch_size)
                X = trainX[start_idx: end_idx]
                TE = trainTE[start_idx: end_idx]
                label = trainY[start_idx: end_idx]
                X, TE = X.to(device), TE.to(device)
                label = label.to(device)

                cuda = True if torch.cuda.is_available() else False
                TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
                valid = torch.autograd.Variable(
                    TensorFloat(self.args.batch_size * (self.args.lag + self.args.num_his), 1).fill_(1.0), requires_grad=False)
                fake = torch.autograd.Variable(
                    TensorFloat(self.args.batch_size * (self.args.lag + self.args.num_his), 1).fill_(0.0), requires_grad=False)

                # -------------------------------------------------------------------
                # Train Generator
                # -------------------------------------------------------------------
                self.optimizer_G.zero_grad()

                output = self.generator(X, TE,self.norm_dis_matrix)


                output = output * std + mean

                fake_input = torch.cat((X, (output - output.mean()) / output.std()),
                                       dim=1) if self.args.real_value else torch.cat((X, output),
                                                                                     dim=1)  # [B'', W, N, 1] // [B'', H, N, 1] -> [B'', W+H, N, 1]
                true_input = torch.cat((X, (label - label.mean()) / label.std()),
                                       dim=1) if not self.args.real_value else torch.cat((X, label), dim=1)

                if self.args.is_GAN:
                    loss_G = masked_mae(output.cuda(), label) + self.args.loss_G_D * self.loss_D(
                        self.discriminator(fake_input),
                        valid)
                else:
                    loss_G = masked_mae(output.cuda(), label, null_val=0.0)
                loss_G.backward()

                # Discriminator
                # -------------------------------------------------------------------
                # Train Discriminator
                # -------------------------------------------------------------------
                self.optimizer_D.zero_grad()
                real_loss = self.loss_D(self.discriminator(true_input), valid)
                fake_loss = self.loss_D(self.discriminator(fake_input.detach()), fake)
                loss_D = 0.5 * (real_loss + fake_loss)
                gp = gradient_penalty(true_input, fake_input, self.discriminator)
                # 将梯度惩罚添加到判别器损失中
                loss_D += self.args.lambda_gp * gp
                loss_D.backward()
                self.optimizer_D.step()
                total_loss_D += loss_D.item()

                # add max grad clipping
                if self.args.grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.max_grad_norm)
                self.optimizer_G.step()
                total_loss_G += loss_G.item()
                torch.cuda.empty_cache()



                # log information
                if batch_idx % self.args.log_step == 0:
                    self.logger.info(
                        'Train Epoch {}: {}/{} Generator Loss: {:.6f} Pred Discriminator Loss: {:.6f}'.format(
                            epoch,
                            batch_idx, train_num_batch,
                            loss_G.item(), loss_D.item()))


            # 用于训练最后一批的其余代码
            if num_train % self.args.batch_size != 0:
                cuda = True if torch.cuda.is_available() else False
                TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
                start_idx = (train_num_batch - 1) * self.args.batch_size
                end_idx = num_train
                remaining_samples = end_idx - start_idx

                X = trainX[start_idx: end_idx]
                TE = trainTE[start_idx: end_idx]
                label = trainY[start_idx: end_idx]

                X, TE = X.to(device), TE.to(device)
                label = label.to(device)

                # Create valid and fake tensors based on the size of remaining samples
                valid = torch.autograd.Variable(
                    TensorFloat(remaining_samples * (self.args.lag + self.args.num_his), 1).fill_(1.0),
                    requires_grad=False)
                fake = torch.autograd.Variable(
                    TensorFloat(remaining_samples * (self.args.lag + self.args.num_his), 1).fill_(0.0),
                    requires_grad=False)
                # Train Generator
                self.optimizer_G.zero_grad()
                output = self.generator(X, TE,self.norm_dis_matrix)
                output = output * std + mean
                fake_input = torch.cat((X, (output - output.mean()) / output.std()),
                                       dim=1) if self.args.real_value else torch.cat((X, output), dim=1)
                true_input = torch.cat((X, (label - label.mean()) / label.std()),
                                       dim=1) if not self.args.real_value else torch.cat((X, label), dim=1)

                if self.args.is_GAN :
                    loss_G = masked_mae(output.cuda(), label) + self.args.loss_G_D * self.loss_D(self.discriminator(fake_input),
                                                                                            valid)
                else:
                    loss_G = masked_mae(output.cuda(), label, null_val=0.0)
                loss_G.backward()

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_loss = self.loss_D(self.discriminator(true_input), valid)
                fake_loss = self.loss_D(self.discriminator(fake_input.detach()), fake)
                loss_D = 0.5 * (real_loss + fake_loss)
                loss_D.backward()
                self.optimizer_D.step()

                total_loss_G += loss_G.item()
                total_loss_D += loss_D.item()

            torch.cuda.empty_cache()

            train_epoch_loss_G = total_loss_G / train_num_batch  # average generator loss
            train_epoch_loss_D = total_loss_D / train_num_batch  # average discriminator loss

            # learning rate decay
            if self.args.lr_decay:
                self.lr_scheduler_G.step()
                self.lr_scheduler_D.step()



            # val_loss
            self.generator.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_idx in range(val_num_batch):
                    start_idx = batch_idx * self.args.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.args.batch_size)
                    X = valX[start_idx: end_idx]
                    TE = valTE[start_idx: end_idx]
                    label = valY[start_idx: end_idx]
                    X, TE = X.to(device), TE.to(device)
                    label = label.to(device)
                    output = self.generator(X, TE,self.norm_dis_matrix)
                    output = output * std + mean
                    loss_batch = masked_mae(output.cuda(), label)
                    total_val_loss += loss_batch * (end_idx - start_idx)
                    torch.cuda.empty_cache()
            val_epoch_loss = total_val_loss / num_val
            record_loss(loss_file, val_epoch_loss)


            # self.lr_scheduler_D_RF.step()
            self.logger.info(
                '**********Train Epoch {}: Averaged Generator Loss: {:.6f}, Averaged Pred Discriminator Loss: {:.6f},val_loss: {:.6f}'.format(
                    epoch,
                    train_epoch_loss_G,
                    train_epoch_loss_D,
                    val_epoch_loss
                ))

            if train_epoch_loss_G > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False

            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs! Training stops!".format(
                        self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.generator.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # save the best model to file
        # if not self.args.debug:
        # test
        self.generator.load_state_dict(best_model)
        self.save_checkpoint()
        # self.val_epoch(self.args.epochs, self.test_loader)
        # self.test(self.generator, self.args, self.logger)

        # 测试

        model_state_dict = torch.load(self.best_path)['state_dict']
        self.generator.load_state_dict(model_state_dict)
        print("Load saved model")
        self.test(self.generator, self.args, self.logger,self.norm_dis_matrix)
        # 关闭日志记录器
        logging.shutdown()

    def save_checkpoint(self):
        state = {
            'state_dict': self.generator.state_dict(),
            'optimizer': self.optimizer_G.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, logger, norm_dis_matrix=None,path=None):
        if path is not None:
            check_point = torch.load(os.path.join(path, 'best_mod1el.pth'))
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)

        (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
         testY, mean, std) = load_data(args)
        num_test = testX.shape[0]
        test_num_batch = math.ceil(num_test / args.batch_size)
        model.eval()
        y_pred = []


        with torch.no_grad():
            for batch_idx in range(test_num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
                X = testX[start_idx: end_idx]
                TE = testTE[start_idx: end_idx]
                X, TE = X.to(device), TE.to(device)
                pred_batch = model(X, TE,norm_dis_matrix)
                y_pred.append(pred_batch.cpu().detach().clone())
                del X, TE, pred_batch

        y_pred = torch.from_numpy(np.concatenate(y_pred, axis=0))
        y_pred = y_pred * std + mean

        # 计算指标
        # each horizon point
        for t in range(testY.shape[1]):  # H
            mae, rmse, mape = metric(y_pred[:, t, ...], testY[:, t, ...])
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100))
        test_mae, test_rmse, test_mape = metric(y_pred, testY)

        logger.info('test             mae %.2f\t\trmse %.2f\t\tmape %.2f%%' %
                    (test_mae, test_rmse, test_mape * 100))
