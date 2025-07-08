# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import torch

from etchat.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_MATCH_TOKEN, IMAGE_TOKEN_INDEX

import numpy as np
from nncore.ops import temporal_iou

from scipy.signal import savgol_filter

def temporal_nms(segments, iou_threshold=0.5):
    # flatten 结构（如果是 list of list of list）
    if isinstance(segments[0][0], (list, tuple, np.ndarray)):
        segments = [s for group in segments for s in group]

    # 转为 tensor，确保兼容 torch ops
    if not isinstance(segments, torch.Tensor):
        segments = torch.tensor(segments, dtype=torch.float32)

    if segments.ndim != 2 or segments.shape[1] != 2:
        raise ValueError("Each segment must be [start, end]")

    if segments.shape[0] == 0:
        return []

    lengths = segments[:, 1] - segments[:, 0]
    order = torch.argsort(lengths, descending=True).flatten()
    keep = []

    while order.numel() > 0:
        i = order[0].item()

        # 新增：跳过长度小于 1 的片段
        if lengths[i] < 2:
            order = order[1:]
            continue

        keep.append(i)

        current = segments[i:i+1]
        remaining = segments[order]

        ious = temporal_iou(remaining, current).squeeze()
        suppressed = (ious <= iou_threshold).nonzero(as_tuple=False).squeeze()
        order = order[suppressed].flatten()

    return keep


def truncate_after_nth_token(text, token, n):
    # 找到第 n 个 token 的位置
    parts = text.split(token)
    if len(parts) <= n:
        return text  # 不足 n 次，不截断
    else:
        # 保留前 n 个 token 和它们之间的内容
        return token.join(parts[:n]) + token


def tokenize(text, tokenizer, is_multimodal=True):
    if not is_multimodal:
        return tokenizer(text, return_tensors='pt').input_ids[0]

    chunks = [tokenizer(c).input_ids for c in text.split(DEFAULT_IMAGE_TOKEN)]

    input_ids, offset = [], 0
    if len(chunks) > 0 and len(chunks[0]) > 0 and chunks[0][0] == tokenizer.bos_token_id:
        input_ids.append(chunks[0][0])
        offset = 1

    img_token = [IMAGE_TOKEN_INDEX] * (offset + 1)
    chunks = [e[offset:] for c in zip(chunks, [img_token] * len(chunks)) for e in c][:-1]

    for chunk_ids in chunks:
        input_ids.extend(chunk_ids)

    input_ids = torch.LongTensor(input_ids)
    return input_ids


def detokenize(tokens, model, tokenizer, template=None):
    text = tokenizer.decode(tokens, skip_special_tokens=False).strip()

    tgt = getattr(model, 'tgt', None)
    sim = getattr(model, 'sim', None)
    task = getattr(model, 'task', None)
    if tgt is not None:
        assert len(tokens) == len(tgt), (tokens, tgt)

        if not torch.is_tensor(tokens):
            tokens = torch.LongTensor(tokens)

        model.match_inds = torch.where(tokens == model.config.match_token_id)[0].tolist()
        tgt = [tgt[i] for i in model.match_inds]

        sim = [sim[i] for i in model.match_inds]

        tgt_detokenize = []

        indices = None

        for sim_i in sim:
            if task not in ('vhd', 'tvc'):
                sim_i = savgol_filter(sim_i, window_length=3, polyorder=2)
            indices = [j for j, sim_i_j in enumerate(sim_i) if sim_i_j > 1e-5]

            if indices is not None:
                tgt_detokenize.append(indices[0])
                tgt_detokenize.append(indices[-1])
            else:
                tgt_detokenize.append(0)
                tgt_detokenize.append(len(tgt) - 1)

        assert text.count(DEFAULT_MATCH_TOKEN) == len(tgt), (text, tgt)
        text = text.replace('{', '{{').replace('}', '}}')
        if task not in ('vhd', 'tvc'):
            if task == 'tal':
                ### Merge overlapped pred_segments
                all_pred_segs = []
                for i in range(len(tgt_detokenize) // 2):
                    all_pred_segs.append((tgt_detokenize[i * 2], tgt_detokenize[i * 2 + 1]))
                
                keep_idx = temporal_nms(all_pred_segs, iou_threshold=0.85)
                filtered_segs = [all_pred_segs[i] for i in keep_idx]
                flattened_segs = [seg for group in filtered_segs for seg in group]

                text = truncate_after_nth_token(text, DEFAULT_MATCH_TOKEN, len(flattened_segs)//2)
                tgt_detokenize = flattened_segs
                
            
            text = text.replace(DEFAULT_MATCH_TOKEN, '{} - {}' if template is None else template)

            text = text.format(*tgt_detokenize)
        else:
            text = text.replace(DEFAULT_MATCH_TOKEN, '{}' if template is None else template)
            text = text.format(*tgt)
    else:
        model.match_inds = None

    return text
