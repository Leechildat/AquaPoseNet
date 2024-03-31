import logging
import time
import os

import numpy as np
import torch

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        input = input.cuda()
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        combined_hm_preds = model(input).cuda()
        output = combined_hm_preds[:,-1]
        if type(combined_hm_preds)!=list and type(combined_hm_preds)!=tuple:
            combined_hm_preds = [combined_hm_preds]

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        # loss = criterion(output, target, target_weight)
        loss = model.calc_loss(combined_hm_preds=combined_hm_preds, heatmaps=target)
        if config.LOSS.COMBINE == 'deth':
            loss_weight = [j for j in range(1, config.MODEL.NUM_STACKS+1)]
            t_mean = sum(loss_weight)/config.MODEL.NUM_STACKS
            for wi in range(len(loss_weight)):
                loss_weight[wi] = loss_weight[wi]/t_mean
            for lossi in range(loss.shape[0]):
                for j in range(config.MODEL.NUM_STACKS):
                    loss[lossi][j] = loss[lossi][j]*loss_weight[j]
        loss = torch.mean(loss)
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)
def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 4),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            input = input.cuda()

            combined_hm_preds = model(input).cuda()
            output = combined_hm_preds[:,-1]
            if type(combined_hm_preds)!=list and type(combined_hm_preds)!=tuple:
                combined_hm_preds = [combined_hm_preds]

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = model.calc_loss(combined_hm_preds=combined_hm_preds, heatmaps=target)
            loss = torch.mean(loss)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            if config.DATASET.DATASET == 'posetrack':
                filenames.extend(meta['filename'])
                imgnums.extend(meta['imgnum'].numpy())

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)
                                  
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path, 
            filenames, imgnums)

        _, full_arch_name = get_model_name(config)
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(name_values, full_arch_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def trainNet(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        input = input.cuda()
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        output = model(input).cuda()
        
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        loss = criterion(output, target, target_weight)
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i) % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

def validateNet(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 4),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            input = input.cuda()

            output = model(input).cuda()

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)
            
            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            if config.DATASET.DATASET == 'posetrack':
                filenames.extend(meta['filename'])
                imgnums.extend(meta['imgnum'].numpy())

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)
                                  
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path, 
            filenames, imgnums)

        _, full_arch_name = get_model_name(config)
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(name_values, full_arch_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            # writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# def train_withWeight(config, train_loader, model, criterion, optimizer, epoch,
#           output_dir, tb_log_dir, writer_dict):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     losses1 = AverageMeter()
#     losses2 = AverageMeter()
#     acc = AverageMeter()
#
#     # switch to train mode
#     model.train()
#
#     end = time.time()
#     for i, (input, target, target_weight, meta, weights_gt) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         # compute output
#         output, weights_output = model(input)
#         target = target.cuda(non_blocking=True)
#         target_weight = target_weight.cuda(non_blocking=True)
#
#         loss1 = criterion(output, target, target_weight)
#
# # TODO: loss选择
#         alpha = 0.0015
#         weights_gt = weights_gt.cuda(non_blocking=True)
#         loss_weights = torch.nn.MSELoss()
#         loss2 = loss_weights(weights_output, weights_gt)
#
#         loss = loss1 + alpha * loss2
#
#         # compute gradient and do update step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # measure accuracy and record loss
#         losses1.update(loss1.item(), input.size(0))
#         losses2.update(loss2.item(), input.size(0))
#         losses.update(loss.item(), input.size(0))
#
#         _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
#                                          target.detach().cpu().numpy())
#         acc.update(avg_acc, cnt)
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % config.PRINT_FREQ == 0:
#             msg = 'Epoch: [{0}][{1}/{2}]\t' \
#                   'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
#                   'Speed {speed:.1f} samples/s\t' \
#                   'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
#                   'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
#                   'Loss1 {loss1.val:.5f} ({loss1.avg:.5f})\t' \
#                   'Loss2 {loss2.val:.5f} ({loss2.avg:.5f})\t'\
#                   'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
#                       epoch, i, len(train_loader), batch_time=batch_time,
#                       speed=input.size(0)/batch_time.val,
#                       data_time=data_time, loss=losses, loss1=losses1, loss2=losses2, acc=acc)
#             logger.info(msg)
#
#             writer = writer_dict['writer']
#             global_steps = writer_dict['train_global_steps']
#             writer.add_scalar('train_loss', losses.val, global_steps)
#             writer.add_scalar('train_loss1', losses1.val, global_steps)
#             writer.add_scalar('train_loss2', losses2.val, global_steps)
#             writer.add_scalar('train_acc', acc.val, global_steps)
#             writer_dict['train_global_steps'] = global_steps + 1
#
#             prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
#             save_debug_images(config, input, meta, target, pred*4, output,
#                               prefix)

# def validate_withWeight(config, val_loader, val_dataset, model, criterion, output_dir,
#              tb_log_dir, writer_dict=None):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     losses1 = AverageMeter()
#     losses2 = AverageMeter()
#     acc = AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#
#     num_samples = len(val_dataset)
#     all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 4),
#                          dtype=np.float32)
#     all_boxes = np.zeros((num_samples, 6))
#     image_path = []
#     filenames = []
#     imgnums = []
#     idx = 0
#     with torch.no_grad():
#         end = time.time()
#         for i, (input, target, target_weight, meta, weights_gt) in enumerate(val_loader):
#             # compute output
#             output, weights_output = model(input)
#             if config.TEST.FLIP_TEST:
#                 # this part is ugly, because pytorch has not supported negative index
#                 # input_flipped = model(input[:, :, :, ::-1])
#                 input_flipped = np.flip(input.cpu().numpy(), 3).copy()
#                 input_flipped = torch.from_numpy(input_flipped).cuda()
#                 output_flipped = model(input_flipped)
#                 output_flipped = flip_back(output_flipped.cpu().numpy(),
#                                            val_dataset.flip_pairs)
#                 output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
#
#                 # feature is not aligned, shift flipped heatmap for higher accuracy
#                 if config.TEST.SHIFT_HEATMAP:
#                     output_flipped[:, :, :, 1:] = \
#                         output_flipped.clone()[:, :, :, 0:-1]
#                     # output_flipped[:, :, :, 0] = 0
#
#                 output = (output + output_flipped) * 0.5
#
#             target = target.cuda(non_blocking=True)
#             target_weight = target_weight.cuda(non_blocking=True)
#
#             loss1 = criterion(output, target, target_weight)
#
#                 # TODO: loss选择
#             alpha = 0.0015
#             weights_gt = weights_gt.cuda(non_blocking=True)
#             loss_weights = torch.nn.MSELoss()
#             loss2 = loss_weights(weights_output, weights_gt)
#
#             loss = loss1 + alpha * loss2
#
#             num_images = input.size(0)
#             # measure accuracy and record loss
#             losses.update(loss.item(), num_images)
#             losses1.update(loss1.item(), num_images)
#             losses2.update(loss2.item(), num_images)
#             _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
#                                              target.cpu().numpy())
#
#             acc.update(avg_acc, cnt)
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             c = meta['center'].numpy()
#             s = meta['scale'].numpy()
#             score = meta['score'].numpy()
#             preds_weights = weights_output.cpu().numpy()
#
#             preds, maxvals = get_final_preds(
#                 config, output.clone().cpu().numpy(), c, s)
#
#             all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
#             all_preds[idx:idx + num_images, :, 2:3] = maxvals
#             all_preds[idx:idx + num_images, :, 3] = preds_weights
#             # double check this all_boxes parts
#             all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
#             all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
#             all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
#             all_boxes[idx:idx + num_images, 5] = score
#             image_path.extend(meta['image'])
#             if config.DATASET.DATASET == 'posetrack':
#                 filenames.extend(meta['filename'])
#                 imgnums.extend(meta['imgnum'].numpy())
#
#             idx += num_images
#
#             if i % config.PRINT_FREQ == 0:
#                 msg = 'Test: [{0}/{1}]\t' \
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
#                       'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t' \
#                       'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'\
#                       'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
#                           i, len(val_loader), batch_time=batch_time,
#                           loss=losses, loss1=losses1,
#                           loss2=losses2, acc=acc)
#                 logger.info(msg)
#
#                 prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
#                 save_debug_images(config, input, meta, target, pred*4, output,
#                                   prefix)
#
#         name_values, perf_indicator = val_dataset.evaluate(
#             config, all_preds, output_dir, all_boxes, image_path,
#             weights_output.clone().cpu().numpy(),
#             filenames, imgnums)
#
#         _, full_arch_name = get_model_name(config)
#         if isinstance(name_values, list):
#             for name_value in name_values:
#                 _print_name_value(name_value, full_arch_name)
#         else:
#             _print_name_value(name_values, full_arch_name)
#
#         if writer_dict:
#             writer = writer_dict['writer']
#             global_steps = writer_dict['valid_global_steps']
#             writer.add_scalar('valid_loss', losses.avg, global_steps)
#             writer.add_scalar('valid_acc', acc.avg, global_steps)
#             writer.add_scalar('valid_loss1', losses1.avg, global_steps)
#             writer.add_scalar('valid_loss2', losses2.avg, global_steps)
#             if isinstance(name_values, list):
#                 for name_value in name_values:
#                     writer.add_scalars('valid', dict(name_value), global_steps)
#             else:
#                 writer.add_scalars('valid', dict(name_values), global_steps)
#             writer_dict['valid_global_steps'] = global_steps + 1
#
#     return perf_indicator

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
