import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchinfo
from sklearn import metrics
from tqdm import tqdm

from configs.schedulers import WarmupMultiStepLR
from models import MyNet
from utils.classification.dataset_getter import get_dataset
from utils.io import weights_saver, json_saver
from utils.runtime import makedirs, update, async_exec


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, local_rank, world_size):
    model.eval()
    val_loader_ = tqdm(val_loader) if local_rank == 0 else val_loader
    total_loss, num_samples = 0., len(val_loader.sampler)
    total, correct = 0, 0
    preds_all, targets_all = [], []

    for images, targets in val_loader_:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        _, preds = outputs.max(1)

        dist.all_reduce(loss)
        loss = loss.item() / world_size
        total_loss += loss * len(targets)

        preds_per_gpu = [torch.zeros_like(preds, dtype=torch.int64, device=device) for _ in range(world_size)]
        targets_per_gpu = [torch.zeros_like(targets, dtype=torch.int64, device=device) for _ in range(world_size)]
        dist.all_gather(preds_per_gpu, preds)
        dist.all_gather(targets_per_gpu, targets)
        for i in preds_per_gpu:
            preds_all.extend(i.cpu().numpy().tolist())
        for i in targets_per_gpu:
            targets_all.extend(i.cpu().numpy().tolist())

        total_per_batch = torch.tensor(len(targets), device=device)

        dist.all_reduce(total_per_batch)
        total += total_per_batch.item()

        correct_per_batch = (preds == targets).sum()
        dist.all_reduce(correct_per_batch)
        correct += correct_per_batch.item()

        acc = correct / total

        val_loader_.desc = f"valid  loss: {loss:.3f} | acc: {acc:.3f}"

    accuracy = metrics.accuracy_score(targets_all, preds_all)
    recall = metrics.recall_score(targets_all, preds_all, average='macro')
    precision = metrics.precision_score(targets_all, preds_all, average='macro')
    f1_score = metrics.f1_score(targets_all, preds_all, average='macro')

    return total_loss / num_samples, accuracy, recall, precision, f1_score


def train(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device, to_save, freeze_bn, **kwargs):
    results = {"loss": [], "accuracy": [], "recall": [], "precision": [], "f1_score": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        if freeze_bn:
            model.module.freeze_bn()
        train_loader.sampler.set_epoch(epoch)
        train_loader_ = tqdm(train_loader) if args.local_rank == 0 else train_loader
        total, correct = 0, 0

        for images, targets in train_loader_:
            images = images.to(device)  # [B, C, H, W]
            targets = targets.to(device)  # [B]

            outputs = model(images)
            loss = criterion(outputs, targets)
            update(model, loss, optimizer)

            _, preds = outputs.max(1)
            total += targets.shape[0]
            correct += (preds == targets).sum().item()
            acc_train = correct / total

            train_loader_.desc = f"epoch [{epoch}/{args.epochs}]  " \
                                 f"loss: {loss.item():.3f} | " \
                                 f"acc: {acc_train:.3f} | " \
                                 f"lr: {optimizer.param_groups[0]['lr']:.2e}"

        scheduler.step()

        if epoch % args.eval_interval == 0:
            loss, accuracy, recall, precision, f1_score = evaluate(model, val_loader, criterion, device,
                                                                   args.local_rank, args.world_size)

            if args.local_rank == 0:
                results["loss"].append(loss)
                results["accuracy"].append(accuracy)
                results["recall"].append(recall)
                results["precision"].append(precision)
                results["f1_score"].append(f1_score)

            if args.local_rank == 0:
                print()

            if to_save and epoch % args.save_interval == 0 and args.local_rank == 0:
                # pth = f"{args.dataset}-{epoch}.pth"
                pth = f"{args.dataset}-latest.pth"
                pth_path = os.path.join(args.save_root, pth)
                async_exec(weights_saver, args=(model, pth_path, args.save_pos), kwargs=dict(data_parallel=True))
                print(f"model saved | ", end="")

                res_path = f"./res-{args.batch_size}.json"
                async_exec(json_saver, args=(results, res_path))

            if args.local_rank == 0:
                print(f"val loss: {loss:.3f}"
                      f" | val acc: {accuracy * 100:.3f}"
                      f" | val R: {recall * 100:.3f}"
                      f" | val P: {precision * 100:.3f}"
                      f" | val f1: {f1_score * 100:.3f}")

        if args.local_rank == 0:
            print(f"dataset: {args.dataset} | batch size: {args.batch_size}\n" +
                  (f"saving path: {os.path.join(args.save_root, f'{args.dataset}-*.pth')}\n" if to_save else "") +
                  (f"remarks: {args.remarks}\n\n" if args.remarks is not None else "\n\n"))


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = not args.cudnn_benchmark
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])

    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    args.world_size = dist.get_world_size()

    args.dataset = args.dataset.lower()
    if args.weights is None:
        pass

    to_save = False if args.do_not_save or args.local_rank != 0 else True
    makedirs(args.save_root, exist_ok=True)

    train_dataset, val_dataset = get_dataset(args.data_root, args.dataset, mode="train")

    if args.local_rank == 0:
        print(f"device: {device}\n"
              f"Using {num_workers} dataloader workers every process\n\n"
              f"dataset path: {os.path.join(args.data_root, args.dataset)}\n" +
              (f"weights path: {args.weights}\n" if args.weights != "" else "") +
              (f" saving root: {args.save_root}\n\n" if to_save else "\n") +
              f"dataset: {args.dataset} | batch size: {args.batch_size}\n" +
              (f"remarks: {args.remarks}\n\n" if args.remarks is not None else "\n"))

    train_sampler = Data.distributed.DistributedSampler(dataset=train_dataset, shuffle=True)
    val_sampler = Data.distributed.DistributedSampler(dataset=val_dataset, shuffle=False)

    train_loader = Data.DataLoader(dataset=train_dataset,
                                   sampler=train_sampler,
                                   batch_size=args.batch_size,
                                   pin_memory=True,
                                   num_workers=num_workers,
                                   collate_fn=train_dataset.collate_fn)

    val_loader = Data.DataLoader(dataset=val_dataset,
                                 sampler=val_sampler,
                                 batch_size=args.batch_size,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=val_dataset.collate_fn)

    model = MyNet()
    model.to(device)

    torchinfo.summary(model, input_data=torch.randn([1, 3, 32, 32]))

    if args.local_rank == 0:
        print("\n")

    if args.sync_bn and not args.freeze_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9,
                          weight_decay=1e-4)
    # optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = WarmupMultiStepLR(optimizer, [60, 120, 200, 260], gamma=0.1, warmup_iters=1, warmup_factor=0.01)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr_lambda(args.epochs, args.lrf))

    kwargs = dict(args=args, model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion,
                  optimizer=optimizer, scheduler=scheduler, device=device, to_save=to_save, freeze_bn=args.freeze_bn)

    train(**kwargs)

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # runtime
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--cudnn-benchmark', type=bool, default=True)
    parser.add_argument('--sync-bn', type=bool, default=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--eval-interval', type=int, default=1)

    # model
    parser.add_argument('--pos', type=str, help='model pos')

    # dataset
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')

    # pre-trained weights
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--load-pos', type=str)
    parser.add_argument('--freeze-bn', action='store_true')

    # saving weights
    parser.add_argument('--do-not-save', action='store_true')
    parser.add_argument('--save-root', type=str, default='./runs/weights')
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--save-pos', type=str)

    # remarks
    parser.add_argument('--remarks', type=str)

    args = parser.parse_args()

    main(args)
