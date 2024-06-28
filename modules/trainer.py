import os
from abc import abstractmethod
from tqdm import tqdm
import time
import torch
import pandas as pd
from numpy import inf
import matplotlib.pyplot as plt


# from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args
        self.model = model
        self.start_epoch = 1
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.epochs = self.args.epochs
        self.save_period = self.args.save_period
        self.mnt_mode = args.monitor_mode

        if self.args.n_gpu > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpus  # select multiple GPUs
            self.device = torch.device('cuda:0')  # always: 0
            self.model = self.model.to(self.device)
            print("GPUs_Used: {}".format(args.n_gpu))
            if args.resume is not None:  # the position is super important!
                self._resume_checkpoint(args.resume)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpus_id)  # always start with 0  # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2" device_ids=[0, 1] 1 equals to GPU: 2
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu  # 0 1 2 3 select a uni-GPU
            self.device = torch.device('cuda:0')  # always: 0
            self.model = self.model.to(self.device)
            if args.resume is not None:
                self._resume_checkpoint(args.resume)  # the position is important!
        # n_gpu_available = torch.cuda.device_count()  # After the os.environ
        # print("GPUs_Available: {}".format(n_gpu_available))

        self.mnt_metric_val = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.checkpoint_dir = args.save_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists("".join([args.save_dir, "/logs"])):
            os.mkdir("".join([args.save_dir, "/logs"]))
        # self.writer = SummaryWriter(log_dir="".join([args.save_dir, "/logs"]))

        self.best_recorder = {'val': {self.mnt_metric_val: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}
        self.epochs_recorder = {}

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Resume training from epoch {}".format(self.start_epoch))

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        loss_ = []
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            print("Epoch: {}".format(epoch))
            log = {'epoch': epoch}
            log.update(result)
            loss_.append(log['train_loss'])
            plt.plot(loss_, color='red', label='loss_train')
            plt.title('Train_loss_plot')
            plt.savefig(os.path.join(self.checkpoint_dir, "Train_loss_plot.png"))
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            best = False
            # replacing the mnt_metric_test in this section with mnt_metric_val to monitor the training with validation set
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.mnt_best)

                except KeyError:
                    print("Warning: Metric '{}' is not found. " "performance monitoring is disabled.".format(
                        self.mnt_metric_test))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric_test]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Performance didn\'t improve for {} epochs. Stops.".format(self.early_stop))
                    break
            self._save_checkpoint(epoch, save_best=best)
            self.epochs_recorder.update(log)
            self._print_epochs_to_file()
            self._record_best(log)
            self._print_best_to_file()
            # self._print_best()
        self.writer.close()

    def _save_checkpoint(self, epoch, save_best=False):

        if self.args.n_gpu == 1:
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.mnt_best}

        elif self.args.n_gpu > 1:
            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.mnt_best
            }
            
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))

        if epoch % self.save_period == 0:
            file = os.path.join(self.checkpoint_dir, 'checkpoint_{}.pth'.format(epoch))
            torch.save(state, file)
            print("Saving checkpoint: {} ...".format(file))

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: {} ...".format(best_path))

    def _print_epochs_to_file(self):
        self.epochs_recorder['time'] = time.asctime(time.localtime(time.time()))
        self.epochs_recorder['visual_extractor'] = self.args.visual_extractor
        self.epochs_recorder['sample_method'] = self.args.sample_method
        self.epochs_recorder['seed'] = self.args.seed
        record_path = os.path.join(self.checkpoint_dir, self.args.dataset_name + '_epochs.csv')
        print("record_path : {}".format(record_path))
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        self.epochs_recorder["test_BO_M"] = self.epochs_recorder["test_METEOR"]
        self.epochs_recorder["test_BP_R"] = self.epochs_recorder["test_ROUGE_L"]
        self.epochs_recorder["test_BQ_C"] = self.epochs_recorder["test_CIDEr"]

        # self.epochs_recorder["val_BO_M"] = self.epochs_recorder["val_METEOR"]
        # self.epochs_recorder["val_BP_R"] = self.epochs_recorder["val_ROUGE_L"]
        # self.epochs_recorder["val_BQ_C"] = self.epochs_recorder["val_CIDEr"]

        record_table = record_table.append(self.epochs_recorder, ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _print_best_to_file(self):
        record_path = os.path.join(self.checkpoint_dir, self.args.dataset_name + '_best.csv')
        print("record_path : {}".format(record_path))
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder, ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _record_best(self, log):
        # improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric_val] <= self.best_recorder['val'][
        #     self.mnt_metric_val]) or \
        #                (self.mnt_mode == 'max' and log[self.mnt_metric_val] >= self.best_recorder['val'][self.mnt_metric_val])
        # if improved_val:
        #     self.best_recorder['val'].update(log)
        #     print("improved_val")

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)
            print("improved_test")

    # def _print_best(self):
    #     print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
    #     for key, value in self.best_recorder['val'].items():
    #         print('\t{:15s}: {}'.format(str(key), value))
    #
    #     print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
    #     for key, value in self.best_recorder['test'].items():
    #         print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()
        with tqdm(desc='Epoch %d - Training' % epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(
                    self.device)
                output = self.model(images, reports_ids, reports_masks)
                loss = self.criterion(output, reports_ids, reports_masks)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                pbar.set_postfix(loss=train_loss / (batch_idx + 1))  # loss per batch
                pbar.update()
            log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.writer.add_scalar('train_loss', train_loss / len(self.train_dataloader), epoch)

        # self.model.eval()
        # with torch.no_grad():
        #     val_gts, val_res = [], []
        #     with tqdm(desc='Epoch %d - Evaluation' % epoch, unit='it', total=len(self.val_dataloader)) as pbar:
        #         for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
        #             images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
        #                 self.device), reports_masks.to(self.device)
        #             output = self.model(images, mode='sample')
        #             if self.args.n_gpu > 1:
        #                 reports = self.model.module.tokenizer.decode_batch(output.cpu().numpy())
        #                 ground_truths = self.model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
        #             else:
        #                 reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
        #                 ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
        #             val_res.extend(reports)
        #             val_gts.extend(ground_truths)
        #             pbar.update()
        #             # i = 0
        #             # for id in images_id:
        #             #     print('\n')
        #             #     print(id)
        #             #     print('Pred Sent.{}'.format(reports[i]))
        #             #     print('Real Sent.{}'.format(ground_truths[i]))
        #             #     i = i + 1
        #             #     print('\n')
        #     val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
        #                                    {i: [re] for i, re in enumerate(val_res)})
        #     log.update(**{'val_' + k: v for k, v in val_met.items()})  # used in the _record_best() function and the self.epochs_recorder.update() function
        #     print(val_met)
        #     # self.writer.add_scalar('Val_BLEU_1', val_met["BLEU_1"], epoch)
        #     # self.writer.add_scalar('Val_BLEU_2', val_met["BLEU_2"], epoch)
        #     # self.writer.add_scalar('Val_BLEU_3', val_met["BLEU_3"], epoch)
        #     # self.writer.add_scalar('Val_BLEU_4', val_met["BLEU_4"], epoch)
        #     # self.writer.add_scalar('Val_ROUGE_L', val_met["ROUGE_L"], epoch)
        #     # self.writer.add_scalar('Val_METEOR', val_met["METEOR"], epoch)
        #     # self.writer.add_scalar('Val_CIDEr', val_met["CIDEr"], epoch)

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            with tqdm(desc='Epoch %d - Testing' % epoch, unit='it', total=len(self.test_dataloader)) as pbar:
                for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                    images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                        self.device), reports_masks.to(self.device)
                    output = self.model(images, mode='sample')
                    if self.args.n_gpu > 1:
                        reports = self.model.module.tokenizer.decode_batch(output.cpu().numpy())
                        ground_truths = self.model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    else:
                        reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                        ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    test_res.extend(reports)
                    test_gts.extend(ground_truths)
                    pbar.update()
                    # i = 0
                    # for id in images_id:
                    #     print(id)
                    #     print('Pred Sent.{}'.format(reports[i]))
                    #     print('Real Sent.{}'.format(ground_truths[i]))
                    #     print('\n')
                    #     i = i + 1
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(test_met)
            self.writer.add_scalar('Test_BLEU_1', test_met["BLEU_1"], epoch)
            self.writer.add_scalar('Test_BLEU_2', test_met["BLEU_2"], epoch)
            self.writer.add_scalar('Test_BLEU_3', test_met["BLEU_3"], epoch)
            self.writer.add_scalar('Test_BLEU_4', test_met["BLEU_4"], epoch)
            self.writer.add_scalar('Test_ROUGE_L', test_met["ROUGE_L"], epoch)
            self.writer.add_scalar('Test_METEOR', test_met["METEOR"], epoch)
            self.writer.add_scalar('Test_CIDEr', test_met["CIDEr"], epoch)
            self.lr_scheduler.step()
        return log