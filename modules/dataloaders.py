import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset, BladderSingleImageDataset
import utils


class AHPDataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, drop_last):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        self.drop_last = drop_last

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.483, 0.483, 0.483),
                                     (0.235, 0.235, 0.235))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.483, 0.483, 0.483),
                                     (0.235, 0.235, 0.235))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'mimic_cxr':
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = BladderSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        if split == 'train':
            if args.distributed:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()
                self.sampler = create_sampler(self.dataset, True, num_tasks, global_rank)
            else:
                self.sampler = None
        else:
            if args.distributed:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()
                self.sampler = create_sampler(self.dataset, False, num_tasks, global_rank)
            else:
                self.sampler = None

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'drop_last': self.drop_last,
            'sampler': self.sampler,
            'pin_memory': False,

        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks)

def create_sampler(dataset, shuffle, num_tasks, global_rank):
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                  shuffle=shuffle)
    return sampler

