import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image


from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval, coco_karpathy_retrieval_eval
from data.iu_xray_dataset import iu_xray_train, iu_xray_val
from data.mimic_dataset import mimic_train, mimic_val
from data.pretrain_dataset import pretrain_dataset
from data.vqa_dataset import vqa_dataset
from transform.randaugment import RandomAugment


def create_dataset(dataset, config, min_scale=0.5, max_words=60):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0),
    #                                  interpolation=InterpolationMode.BICUBIC),
    #     transforms.RandomHorizontalFlip(),
    #     RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
    #                                           'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    # transform_test = transforms.Compose([
    #     transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # vqa
    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
    #     transforms.RandomHorizontalFlip(),
    #     RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
    #                                           'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    # transform_test = transforms.Compose([
    #     transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    if dataset == 'pretrain':
        dataset = pretrain_dataset(config['train_file'], config['laion_path'], transform_train)
        return dataset

    elif dataset == "caption_mimic_cxr":
        train_dataset = mimic_train(transform_train, config['image_root'], config['ann_root'], max_words=100)
        val_dataset = mimic_val(transform_test, config['image_root'], config['ann_root'], 'val', max_words=100)
        test_dataset = mimic_val(transform_test, config['image_root'], config['ann_root'], 'test', max_words=100)
        return train_dataset, val_dataset, test_dataset

    elif dataset == "caption_iu_xray":
        train_dataset = iu_xray_train(transform_train, config['image_root'], config['ann_root'], max_words=60)
        val_dataset = iu_xray_val(transform_test, config['image_root'], config['ann_root'], 'val', max_words=60)
        test_dataset = iu_xray_val(transform_test, config['image_root'], config['ann_root'], 'test', max_words=60)
        return train_dataset, val_dataset, test_dataset

    elif dataset=='vqa':
        train_dataset = vqa_dataset(config['train_file'], transform_train, config['vqa_root'], split='train')
        test_dataset = vqa_dataset(config['test_file'], transform_test, config['vqa_root'], split='test',
                                   answer_list=config['answer_list'])
        return train_dataset, test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders

