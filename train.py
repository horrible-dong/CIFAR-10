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
from utils.classification.dataset_getter import get_dataset
from utils.io import weights_saver, json_saver
from utils.runtime import makedirs, update, async_exec


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_features, num_heads, qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.attn = Attention(in_features, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.norm2 = nn.LayerNorm(in_features)
        self.mlp = Mlp(in_features=in_features, drop=drop_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(in_features)
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_features)

    def forward(self, x):
        x_ = x
        x = self.conv(x)
        x += x_
        x = self.relu(x)
        x = self.bn(x)

        return x


class MyNet(nn.Module):
    def __init__(self, c0=3, c1=64, c2=128, c3=256, c4=512, depth=4, heads=4, num_classes=10, drop_ratio=0.,
                 attn_drop_ratio=0.):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(c1),
            ResidualBlock(c1),
            ResidualBlock(c1),
            nn.Conv2d(c1, c2, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(c2),
            ResidualBlock(c2),
            ResidualBlock(c2),
            nn.Conv2d(c2, c3, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(c3),
            ResidualBlock(c3),
            ResidualBlock(c3),
            nn.Conv2d(c3, c4, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(c4),
            ResidualBlock(c4),
            ResidualBlock(c4),
        )

        self.blocks = nn.Sequential(*[
            AttentionBlock(in_features=c4, num_heads=heads, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)
            for _ in range(depth)
        ])

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(c4 * 4, num_classes)

        self.apply(_init_weights)

    def forward(self, x):
        x = self.conv(x)  # [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, H * W, C]

        for block in self.blocks:
            x = block(x)

        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


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
    makedirs(args.save_root)

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
