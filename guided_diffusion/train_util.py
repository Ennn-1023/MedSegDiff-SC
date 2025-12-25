import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
#from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
# from visdom import Visdom
# viz = Visdom(port=8850)
# loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='loss'))
# grad_window = viz.line(Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(),
#                            opts=dict(xlabel='step', ylabel='amplitude', title='gradient'))


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        classifier,
        diffusion,
        data,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.dataloader = dataloader
        self.classifier = classifier
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0

        # 單機單卡版本：global_batch 就是 batch_size 本身
        self.global_batch = self.batch_size

        # 有 GPU 就用 GPU，沒有就用 CPU
        if th.cuda.is_available():
            self.device = dist_util.dev()
            self.model.to(self.device)
        else:
            self.device = th.device("cpu")
            self.model.to(self.device)

        # 是否要在一些地方做 cuda 同步（保留原本邏輯）
        self.sync_cuda = th.cuda.is_available()

        # 載入權重（如果有 resume_checkpoint）
        self._load_and_sync_parameters()

        # 混合精度訓練包裝
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        
        # 打印優化器將要訓練的參數數量
        trainable_params = sum(p.numel() for p in self.mp_trainer.master_params)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.log(f"Optimizer will train {trainable_params:,} / {total_params:,} parameters "
                   f"({100.0 * trainable_params / total_params:.2f}%)")

        # Optimizer
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        # 🔴 完全關掉 DDP / 分散式，直接用單卡模型
        self.use_ddp = False
        self.ddp_model = self.model


    def _load_and_sync_parameters(self):
        """
        單機單卡版：
        - 如果有給 resume_checkpoint，就載入那個權重
        - 不再呼叫 dist.get_rank() 或做 multi-process 同步
        - 支援從非 LoRA checkpoint 載入到 LoRA 模型
        """
        if self.resume_checkpoint:
            # 如果你的檔名有 step，可以解析；沒有也沒關係，失敗就設 0
            try:
                self.resume_step = parse_resume_step_from_filename(
                    self.resume_checkpoint
                )
            except Exception:
                self.resume_step = 0

            logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")
            # 用 dist_util.load_state_dict 幫你處理 CPU 載入
            state_dict = dist_util.load_state_dict(
                self.resume_checkpoint, map_location="cpu"
            )
            
            # 檢測 checkpoint 和模型的 LoRA 狀態
            from .lora import detect_lora_in_state_dict
            checkpoint_lora_info = detect_lora_in_state_dict(state_dict)
            
            # 檢查模型是否有 LoRA
            model_has_lora = any('lora' in name for name, _ in self.model.named_parameters())
            
            # 情況 1: 模型有 LoRA，但 checkpoint 沒有 LoRA（從預訓練模型開始訓練）
            if model_has_lora and not checkpoint_lora_info['has_lora']:
                logger.log("⚠️  Loading non-LoRA checkpoint into LoRA model")
                logger.log("   Remapping weights: 'weight' -> 'linear.weight'")
                
                # 重新映射權重名稱
                new_state_dict = {}
                for key, value in state_dict.items():
                    # 檢查是否是被 LoRA 包裝的層
                    # 例如: input_blocks.1.0.emb_layers.1.weight -> input_blocks.1.0.emb_layers.1.linear.weight
                    if 'emb_layers' in key and (key.endswith('.weight') or key.endswith('.bias')):
                        # 插入 .linear
                        parts = key.rsplit('.', 1)  # 分割最後一個 '.'
                        new_key = parts[0] + '.linear.' + parts[1]
                        new_state_dict[new_key] = value
                        logger.log(f"   Mapped: {key} -> {new_key}")
                    else:
                        new_state_dict[key] = value
                
                state_dict = new_state_dict
                logger.log("✅ Weight remapping completed")
                logger.log("   LoRA parameters will be initialized to zero")
            
            # 情況 2: 兩者都有 LoRA 或都沒有 LoRA
            else:
                if model_has_lora and checkpoint_lora_info['has_lora']:
                    logger.log(f"✅ Loading LoRA checkpoint (rank={checkpoint_lora_info['rank']})")
                elif not model_has_lora and not checkpoint_lora_info['has_lora']:
                    logger.log("✅ Loading standard checkpoint")
            
            # 載入權重（strict=False 允許 LoRA 參數缺失）
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            # 只報告非 LoRA 的缺失鍵
            non_lora_missing = [k for k in missing_keys if 'lora' not in k]
            non_lora_unexpected = [k for k in unexpected_keys if 'lora' not in k]
            
            if non_lora_missing:
                logger.log(f"⚠️  Warning: Missing keys (non-LoRA): {non_lora_missing[:5]}...")
            if non_lora_unexpected:
                logger.log(f"⚠️  Warning: Unexpected keys (non-LoRA): {non_lora_unexpected[:5]}...")
            
            # 再把 model 丟回正確的 device（在 __init__ 裡已經設好 self.device）
            self.model.to(self.device)

        # 單進程情境下，不需要同步參數，原本這行可以拿掉：
        # dist_util.sync_params(self.model.parameters())


    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        i = 0
        data_iter = iter(self.dataloader)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):


            try:
                    batch, cond, name = next(data_iter)
            except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    data_iter = iter(self.dataloader)
                    batch, cond, name = next(data_iter)

            self.run_step(batch, cond)

           
            i += 1
          
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        batch=th.cat((batch, cond), dim=1)

        cond={}
        sample = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return sample

    def forward_backward(self, batch, cond):

        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                self.classifier,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses1[0]["loss"].detach()
                )
            losses = losses1[0]
            sample = losses1[1]

            loss = (losses["loss"] * weights + losses['loss_cal'] * 10).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            for name, param in self.ddp_model.named_parameters():
                if param.grad is None:
                    print(name)
            return  sample

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        """
        單機單卡版本的 checkpoint 存檔：
        - 不使用 dist.get_rank()
        - 不呼叫 dist.barrier()
        - 直接把 model / EMA / optimizer 存到 logger 目前的目錄
        """
        def save_checkpoint(rate, params):
            # 把 master_params 轉回 state_dict
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            
            # 檢測是否使用 LoRA 並添加元數據
            from .lora import detect_lora_in_state_dict
            lora_info = detect_lora_in_state_dict(state_dict)
            if lora_info['has_lora']:
                # 添加 LoRA 元數據（使用特殊 key 不與模型參數衝突）
                state_dict['_lora_config'] = {
                    'rank': lora_info['rank'],
                    'num_layers': lora_info['num_lora_layers'],
                    'has_lora': True
                }
                logger.log(f"  Saving with LoRA config: rank={lora_info['rank']}, layers={lora_info['num_lora_layers']}")

            # 這裡不再判斷 rank，因為只有一個 process
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"

            # 存到 log 目錄底下
            with bf.BlobFile(
                bf.join(get_blob_logdir(), filename), "wb"
            ) as f:
                th.save(state_dict, f)

        # 存主模型
        save_checkpoint(0, self.mp_trainer.master_params)

        # 存 EMA 模型
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # 存 optimizer 狀態
        opt_filename = f"opt{(self.step + self.resume_step):06d}.pt"
        with bf.BlobFile(
            bf.join(get_blob_logdir(), opt_filename), "wb"
        ) as f:
            th.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
