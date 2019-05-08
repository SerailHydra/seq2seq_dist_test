import logging
import math
import time
import os
from itertools import cycle

import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))

import torch
import torch.optim
import torch.utils.data
#from torch._C import start_cupti_tracing, end_cupti_tracing

from seq2seq.train.distributed import DistributedDataParallel as DDP
from seq2seq.train.fp_optimizers import Fp16Optimizer, Fp32Optimizer
from seq2seq.utils import AverageMeter
from seq2seq.utils import sync_workers

import torchmodules.torchprofiler as torchprofiler
from torch.nn.utils import clip_grad_norm_

class Seq2SeqTrainer(object):

    def __init__(self, model, criterion, opt_config,
                 print_freq=10,
                 save_freq=1000,
                 grad_clip=float('inf'),
                 batch_first=False,
                 save_info={},
                 save_path='.',
                 checkpoint_filename='checkpoint%s.pth',
                 keep_checkpoints=5,
                 math='fp32',
                 cuda=True,
                 distributed=False,
                 verbose=False,
                 log_dir=None,
                 num_minibatches=20,
                 cupti=False):
        super(Seq2SeqTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.epoch = 0
        self.save_info = save_info
        self.save_path = save_path
        self.save_freq = save_freq
        self.save_counter = 0
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_counter = cycle(range(keep_checkpoints))
        self.opt_config = opt_config
        self.cuda = cuda
        self.distributed = distributed
        self.print_freq = print_freq
        self.batch_first = batch_first
        self.verbose = verbose
        self.loss = None
        self.cupti = cupti

        self.log_dir = log_dir
        self.num_steps = num_minibatches
        self.math = math
        self.grad_clip = grad_clip

        if cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        if distributed:
            self.model = DDP(self.model, log_dir=self.log_dir)

        if math == 'fp16':
            self.model = self.model.half()
            self.fp_optimizer = Fp16Optimizer(self.model, grad_clip)
            params = self.fp_optimizer.fp32_params
        elif math == 'fp32':
            self.fp_optimizer = Fp32Optimizer(self.model, grad_clip)
            params = self.model.parameters()

        opt_name = opt_config['optimizer']
        lr = opt_config['lr']
        self.optimizer = torch.optim.__dict__[opt_name](params, lr=lr)
        print("optimizer name " + opt_name)
        print(type(self.optimizer))

    def iterate(self, src, tgt, update=True, training=True):
        src, src_length = src
        tgt, tgt_length = tgt
        src_length = torch.LongTensor(src_length)
        tgt_length = torch.LongTensor(tgt_length)

        num_toks = {}
        num_toks['tgt'] = int(sum(tgt_length - 1))
        num_toks['src'] = int(sum(src_length))

        if self.cuda:
            src = src.cuda()
            src_length = src_length.cuda()
            tgt = tgt.cuda()

        t0 = time.time()
        if self.batch_first:
            output = self.model(src, src_length, tgt[:, :-1])
            tgt_labels = tgt[:, 1:]
            T, B = output.size(1), output.size(0)
        else:
            output = self.model(src, src_length, tgt[:-1])
            tgt_labels = tgt[1:]
            T, B = output.size(0), output.size(1)

        loss = self.criterion(output.view(T * B, -1).float(),
                              tgt_labels.contiguous().view(-1))

        loss_per_batch = loss.item()
        loss /= B

        if training:
            self.fp_optimizer.step(loss, self.optimizer, update)

        loss_per_token = loss_per_batch / num_toks['tgt']
        loss_per_sentence = loss_per_batch / B

        return loss_per_token, loss_per_sentence, num_toks

    def feed_data(self, data_loader, training=True):
        if training:
            assert self.optimizer is not None
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_per_token = AverageMeter()
        losses_per_sentence = AverageMeter()

        tot_tok_time = AverageMeter()
        src_tok_time = AverageMeter()
        tgt_tok_time = AverageMeter()

        batch_size = data_loader.batch_size

        end = time.time()
        for i, (src, tgt, _) in enumerate(data_loader):
            print("iteration {}".format(i))
            if i >= self.num_steps and self.num_steps > 0:
                break
            if i == 5 and self.cupti:
                start_cupti_tracing()
            self.save_counter += 1
            # measure data loading time
            data_time.update(time.time() - end)

            # do a train/evaluate iteration
            stats = self.iterate(src, tgt, training=training)
            loss_per_token, loss_per_sentence, num_toks = stats

            # measure accuracy and record loss
            losses_per_token.update(loss_per_token, num_toks['tgt'])
            losses_per_sentence.update(loss_per_sentence, batch_size)

            # measure elapsed time
            elapsed = time.time() - end
            batch_time.update(elapsed)
            src_tok_time.update(num_toks['src'] / elapsed)
            tgt_tok_time.update(num_toks['tgt'] / elapsed)
            tot_num_toks = num_toks['tgt'] + num_toks['src']
            tot_tok_time.update(tot_num_toks / elapsed)
            self.loss = losses_per_token.avg
            end = time.time()

            if i % self.print_freq == 0:
                phase = 'TRAIN' if training else 'EVAL'
                log = []
                log += ['{} [{}][{}/{}]'.format(phase, self.epoch, i, len(data_loader))]
                log += ['Time {:.3f} ({:.3f})'.format(batch_time.val, batch_time.avg)]
                log += ['Data {:.3f} ({:.3f})'.format(data_time.val, data_time.avg)]
                log += ['Tok/s {:.0f} ({:.0f})'.format(tot_tok_time.val, tot_tok_time.avg)]
                if self.verbose:
                    log += ['Src tok/s {:.0f} ({:.0f})'.format(src_tok_time.val, src_tok_time.avg)]
                    log += ['Tgt tok/s {:.0f} ({:.0f})'.format(tgt_tok_time.val, tgt_tok_time.avg)]
                    log += ['Loss/sentence {:.1f} ({:.1f})'.format(losses_per_sentence.val, losses_per_sentence.avg)]
                log += ['Loss/tok {:.8f} ({:.8f})'.format(losses_per_token.val, losses_per_token.avg)]
                log = '\t'.join(log)
                print(log)
                #logging.info(log)

            save_chkpt = (self.save_counter % self.save_freq) == (self.save_freq - 1)
            if training and save_chkpt:
                self.save_counter = 0
                self.save_info['iteration'] = i
                identifier = next(self.checkpoint_counter, -1)
                if identifier != -1:
                    with sync_workers() as rank:
                        if rank == 0:
                            self.save(identifier=identifier)
        if self.cupti:
            end_cupti_tracing()
        return losses_per_token.avg

    def profile_feed_data(self, data_loader, log_dir, training=True):
        pid = os.getpid()
        stdlog = open(os.path.join(log_dir, "stdout.{}.txt".format(pid)), "a+")
        #summary = torchsummary.summary(self.model, , data_loader.batch_size, False)
        if training:
            assert self.optimizer is not None
        if self.math != "fp32":
            print("fp32 only supperted currently")
            return

        fflayer_timestamps = []
        bplayer_timestamps = []
        iteration_timestamps = []
        timestamp_blobs = {}
        timestamp_blobs["optimizer_step"] = []
        timestamp_blobs["ffpass"] = []
        timestamp_blobs["loss"] = []
        timestamp_blobs["bppass"] = []

        if self.cupti:
            start_cupti_tracing()
        for i, (src, tgt, _) in enumerate(data_loader):
            print("iteration {}".format(i))
            if i >= self.num_steps:
                break
            # do a train/evaluate iteration
            # stats = self.iterate(src, tgt, training=training)
            src, src_length = src
            tgt, tgt_length = tgt
            src_length = torch.LongTensor(src_length)
            tgt_length = torch.LongTensor(tgt_length)

            num_toks = {}
            num_toks['tgt'] = int(sum(tgt_length - 1))
            num_toks['src'] = int(sum(src_length))

            t0 = time.time()

            if self.cuda:
                src = src.cuda()
                src_length = src_length.cuda()
                tgt = tgt.cuda()

            with torchprofiler.Profiling(self.model) as p:
                t1 = time.time()
                if self.batch_first:
                    output = self.model(src, src_length, tgt[:, :-1])
                    tgt_labels = tgt[:, 1:]
                    T, B = output.size(1), output.size(0)
                else:
                    output = self.model(src, src_length, tgt[:-1])
                    tgt_labels = tgt[1:]
                    T, B = output.size(0), output.size(1)
                t2 = time.time()
                loss = self.criterion(output.view(T * B, -1).float(),
                                      tgt_labels.contiguous().view(-1))
                loss_per_batch = loss.item()
                loss /= B
                t3 = time.time()

                if not training:
                    continue

                #print(type(self.fp_optimizer))
                #self.fp_optimizer.step(loss, self.optimizer, True)
                self.model.zero_grad()
                loss.backward()
                t4 = time.time()
                if self.grad_clip != float('inf'):
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                t5 = time.time()
            print("iteration time: {}".format(t5 - t0))

            fflayer_timestamps.append(p.processed_times_dir('forward'))
            bplayer_timestamps.append(p.processed_times_dir('backward'))
            timestamp_blobs["optimizer_step"].append({
                "start": t4 * 1000 * 1000,
                "duration": (t5 - t4) * 1000 * 1000,
                "in_critical_path": True,
                "pid": pid})
            timestamp_blobs["ffpass"].append({
                "start": t1 * 1000 * 1000,
                "duration": (t2 - t1) * 1000 * 1000,
                "in_critical_path": True,
                "pid": pid})
            timestamp_blobs["loss"].append({
                "start": t2 * 1000 * 1000,
                "duration": (t3 - t2) * 1000 * 1000,
                "in_critical_path": True,
                "pid": pid})
            timestamp_blobs["bppass"].append({
                "start": t3 * 1000 * 1000,
                "duration": (t4 - t3) * 1000 * 1000,
                "in_critical_path": True,
                "pid": pid})
            iteration_timestamps.append({
                "start": t1 * 1000 * 1000,
                "duration": (t5 - t1) * 1000 * 1000})

        '''
        layer_times = []

        for i in range(len(layer_timestamps[0])):
            layer_type = str(layer_timestamps[0][i][0])
            layer_forward_times = []
            layer_backward_times = []
            for j in range(len(layer_timestamps)):
                layer_forward_times.append(layer_timestamps[j][i][2])
                layer_backward_times.append(layer_timestamps[j][i][5])
            layer_times.append([layer_type, layer_forward_times, layer_backward_times])
        '''

        print("finish iterations")
        if self.cupti:
            end_cupti_tracing()
            print("end cupti tracing")

        torchprofiler.generate_json(0, self.num_steps - 1, fflayer_timestamps, bplayer_timestamps, iteration_timestamps, timestamp_blobs, output_filename=os.path.join(log_dir, "processed_time.{}.json".format(pid)))

        '''
        summary_i = 0
        per_layer_times_i = 0
        while summary_i < len(summary) and per_layer_times_i < len(layer_times):
            summary_elem = summary[summary_i]
            per_layer_time = layer_times[per_layer_times_i]
            if str(summary_elem['layer_name']) != str(per_layer_time[0]):
                summary_elem['forward_time'] = 0.0
                summary_elem['backward_time'] = 0.0
                summary_i += 1
                continue
            summary_elem['forward_time'] = per_layer_time[1]
            summary_elem['backward_time'] = per_layer_time[2]
            summary_i += 1
            per_layer_times_i += 1
        torchgraph.create_graph(model, train_loader, summary, "temp")
        '''

    def preallocate(self, data_loader, training):
        batch_size = data_loader.batch_size
        max_len = data_loader.dataset.max_len

        src_length = [max_len] * batch_size
        tgt_length = [max_len] * batch_size

        if self.batch_first:
            shape = (batch_size, max_len)
        else:
            shape = (max_len, batch_size)

        src = torch.full(shape, 4, dtype=torch.int64)
        tgt = torch.full(shape, 4, dtype=torch.int64)
        src = src, src_length
        tgt = tgt, tgt_length
        self.iterate(src, tgt, update=False, training=training)

    def optimize(self, data_loader):
        torch.set_grad_enabled(True)
        self.model.train()
        torch.cuda.empty_cache()
        self.preallocate(data_loader, training=True)
        if self.log_dir:
            output = self.profile_feed_data(data_loader, self.log_dir, training=True)
        else:
            output = self.feed_data(data_loader, training=True)
        torch.cuda.empty_cache()
        return output

    def evaluate(self, data_loader):
        torch.set_grad_enabled(False)
        self.model.eval()
        torch.cuda.empty_cache()
        self.preallocate(data_loader, training=False)
        output = self.feed_data(data_loader, training=False)
        torch.cuda.empty_cache()
        return output

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location={'cuda:0': 'cpu'})
            self.model.load_state_dict(checkpoint['state_dict'])
            self.fp_optimizer.initialize_model(self.model)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']
            logging.info('loaded checkpoint {} (epoch {})'.format(filename, self.epoch))
        else:
            logging.error('invalid checkpoint: {}'.format(filename))

    def save(self, identifier=None, is_best=False, save_all=False):

        def write_checkpoint(state, filename):
            filename = os.path.join(self.save_path, filename)
            logging.info('saving model to {}'.format(filename))
            torch.save(state, filename)

        state = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': getattr(self, 'loss', None),
        }
        state = dict(list(state.items()) + list(self.save_info.items()))

        if identifier is not None:
            filename = self.checkpoint_filename % identifier
            write_checkpoint(state, filename)

        if is_best:
            filename = 'model_best.pth'
            write_checkpoint(state, filename)

        if save_all:
            filename = 'checkpoint_epoch_{:03d}.pth'.format(self.epoch)
            write_checkpoint(state, filename)
