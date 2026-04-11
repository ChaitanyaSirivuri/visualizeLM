import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers.utils import logging

logger = logging.get_logger(__name__)

SEPARATORS_LIST = [
    '.', ',', '?', '!', ':', ';', '</s>', '/', '(', ')', '[', ']', '{', '}',
    '<', '>', '|', '\\', '-', '_', '+', '=', '*', '&', '^', '%', '$', '#',
    '@', '~', '`', ' ', '\t', '\n', '\r', '\x0b', '\x0c',
]


# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# rule 6 from paper
def handle_self_attention_image(R_i_i, enc_attn_weights, privious_cam=[]):
    device = privious_cam[-1].device if privious_cam else None
    for i, blk in enumerate(enc_attn_weights):
        grad = blk.grad.float().detach()
        cam = blk.float().detach()
        if device is None:
            device = cam.device
        cam = avg_heads(cam.to(device), grad.to(device))
        # rebuild the previous attentions to the same size as the current attention
        if len(privious_cam) != 0 and cam.shape[0] == 1:
            len_seq, all_len_seq = privious_cam[i].shape
            assert len_seq == all_len_seq, "The previous CAMs are not square"
            new_column = torch.zeros(len_seq, 1).to(cam.device)
            privious_cam[i] = torch.cat((privious_cam[i], new_column), dim=1)
            privious_cam[i] = torch.cat((privious_cam[i], cam), dim=0)
            cam = privious_cam[i]
        elif cam.shape[0] != 1:
            privious_cam.append(cam)
        assert cam.shape == R_i_i.shape, "The attention weights and the relevancy map are not the same size"
        R_i_i += torch.matmul(cam, R_i_i)
        del grad, cam

    return R_i_i, privious_cam


def handle_self_attention_image_vit(
    R_i_i_init, enc_attn_weights_vit, img_idx=None,
    add_skip=False, normalize=False, image_seq_length=576,
):
    if img_idx:
        R_i_i = R_i_i_init[img_idx : img_idx + image_seq_length, img_idx : img_idx + image_seq_length]
        if add_skip:
            R_i_i = R_i_i + torch.eye(R_i_i.shape[-1]).to(R_i_i.device)
        # Pad with a zero first row and column for the CLS token.
        R_i_i = torch.cat((torch.zeros(1, R_i_i.shape[1]).to(R_i_i.device), R_i_i), dim=0)
        R_i_i = torch.cat((torch.zeros(R_i_i.shape[0], 1).to(R_i_i.device), R_i_i), dim=1)
        R_i_i[0, 0] = 1
    else:
        R_i_i = R_i_i_init
    if normalize:
        R_i_i = handle_residual(R_i_i)
    for blk_vit in enc_attn_weights_vit:
        grad_vit = blk_vit.grad.float().detach()
        cam_vit = blk_vit.float().detach()
        cam_vit = avg_heads(cam_vit, grad_vit)
        assert cam_vit.shape == R_i_i.shape, "The vit relevancy map and the llama relevancy map are not the same size"
        R_i_i += torch.matmul(cam_vit, R_i_i)
    return R_i_i


# normalization - eq. 8+9
def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    sum_rows = self_attention.sum(dim=-1, keepdim=True)
    sum_rows[sum_rows == 0] = 1
    self_attention = self_attention / sum_rows
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention


def compute_word_rel_map(
    tokens, target_index, R_i_i, separators_list,
    current_rel_map, current_count, current_word, word_rel_maps,
):
    if target_index == 0:
        current_word = tokens[target_index]
        current_rel_map = R_i_i
        current_count = 1
    elif not tokens[target_index].startswith('▁') and tokens[target_index] not in separators_list:
        # Token is part of the current word: accumulate its relevancy.
        current_word += tokens[target_index]
        if current_rel_map.shape[0] < R_i_i.shape[0]:
            padding = (
                0, R_i_i.shape[1] - current_rel_map.shape[1],
                0, R_i_i.shape[0] - current_rel_map.shape[0],
            )
            current_rel_map = F.pad(current_rel_map, padding, "constant", 0)
        current_rel_map += R_i_i
        current_count += 1
    else:
        # Word boundary: store the completed word and start a new one.
        word_rel_maps[current_word] = current_rel_map / current_count
        current_word = tokens[target_index]
        current_rel_map = R_i_i
        current_count = 1
    return word_rel_maps, current_rel_map, current_count, current_word


def construct_relevancy_map(
    tokenizer, model, input_ids, tokens, outputs, output_ids, img_idx,
    apply_normalization=True, image_seq_length=576,
):
    logger.debug('Tokens: %s', tokens)
    enable_vit_relevancy = len(model.enc_attn_weights_vit) > 0
    enc_attn_weights_vit = model.enc_attn_weights_vit if enable_vit_relevancy else None
    enc_attn_weights = model.enc_attn_weights

    num_generated_tokens = len(outputs.attentions)
    num_self_att_layers = len(outputs.attentions[0])
    assert num_generated_tokens == len(outputs.scores)
    assert num_generated_tokens * num_self_att_layers == len(enc_attn_weights), (
        f'{num_generated_tokens}x{num_self_att_layers} != {len(enc_attn_weights)}'
    )
    assert len(tokens) == len(outputs.scores), (
        f'Length of tokens {len(tokens)} is not equal to the length of outputs.scores '
        f'{len(outputs.scores)}\ntokens: {tokens}'
    )

    # Group attention weights by generated token: one sub-list per token.
    enc_attn_weights = [
        enc_attn_weights[i * num_self_att_layers : (i + 1) * num_self_att_layers]
        for i in range(num_generated_tokens)
    ]

    if enable_vit_relevancy:
        enc_attn_weights_vit = enc_attn_weights_vit[:-1]  # last ViT layer skipped for llava
        assert len(enc_attn_weights_vit) > 0

    word_rel_maps_llama, word_rel_maps_all = {}, {}
    word_rel_maps_vit, word_rel_maps_all_generated_token = {}, {}
    privious_cam = []

    current_rel_map = current_rel_map_all = None
    current_rel_map_vit = current_rel_map_all_generated_token = None
    current_word = current_word_all = current_word_vit = current_word_all_generated_token = None
    current_count = current_count_vit = current_count_all = current_count_all_generated_token = 0

    rel_maps_dict = {}
    for target_index in tqdm(range(len(outputs.scores)), desc="Building relevancy maps"):
        token_logits = outputs.scores[target_index]
        token_id = torch.tensor(output_ids[target_index]).to(token_logits.device)

        assert token_id == output_ids[target_index], "token_id does not match output_id"

        token_id_one_hot = torch.nn.functional.one_hot(
            token_id, num_classes=token_logits.size(-1)
        ).float().view(1, -1)
        token_id_one_hot.requires_grad_(True)

        model.zero_grad()
        token_logits.backward(gradient=token_id_one_hot, retain_graph=True)

        seq_len = enc_attn_weights[target_index][0].shape[-1]
        R_i_i_init = torch.eye(seq_len, seq_len).to(token_logits.device).float()
        R_i_i, privious_cam = handle_self_attention_image(
            R_i_i_init, enc_attn_weights[target_index], privious_cam,
        )

        if enable_vit_relevancy:
            R_i_i_all = handle_self_attention_image_vit(
                R_i_i, enc_attn_weights_vit, img_idx,
                add_skip=False, normalize=False, image_seq_length=image_seq_length,
            )

            vit_side = enc_attn_weights_vit[0].shape[-1]
            img_slice = R_i_i[-1, :][img_idx : img_idx + image_seq_length]

            # option #2: seed with last-token image relevancy on the CLS row/column.
            R_i_i_init_vit_all = torch.eye(vit_side, vit_side).to(token_logits.device).float()
            R_i_i_init_vit_all[0, 1:] = R_i_i_init_vit_all[0, 1:] + img_slice
            R_i_i_init_vit_all[1:, 0] = R_i_i_init_vit_all[1:, 0] + img_slice
            R_i_i_all_generated_token = handle_self_attention_image_vit(
                R_i_i_init_vit_all, enc_attn_weights_vit,
            )

            R_i_i_init_vit = torch.eye(vit_side, vit_side).to(token_logits.device).float()
            R_i_i_vit = handle_self_attention_image_vit(R_i_i_init_vit, enc_attn_weights_vit)

        if apply_normalization:
            R_i_i = handle_residual(R_i_i)
            if enable_vit_relevancy:
                R_i_i_all = handle_residual(R_i_i_all)
                R_i_i_vit = handle_residual(R_i_i_vit)
                R_i_i_all_generated_token = handle_residual(R_i_i_all_generated_token)
        else:
            R_i_i = R_i_i - torch.eye(seq_len, seq_len).to(token_logits.device).float()

        # Per-token bucket (disambiguate duplicate tokens by suffixing '_').
        if tokens[target_index] in rel_maps_dict:
            tokens[target_index] = tokens[target_index] + '_'
        rel_maps_dict[tokens[target_index]] = R_i_i

        word_rel_maps_llama, current_rel_map, current_count, current_word = compute_word_rel_map(
            tokens, target_index, R_i_i, SEPARATORS_LIST,
            current_rel_map, current_count, current_word, word_rel_maps_llama,
        )

        if enable_vit_relevancy:
            word_rel_maps_all, current_rel_map_all, current_count_all, current_word_all = \
                compute_word_rel_map(
                    tokens, target_index, R_i_i_all, SEPARATORS_LIST,
                    current_rel_map_all, current_count_all, current_word_all, word_rel_maps_all,
                )
            word_rel_maps_vit, current_rel_map_vit, current_count_vit, current_word_vit = \
                compute_word_rel_map(
                    tokens, target_index, R_i_i_vit, SEPARATORS_LIST,
                    current_rel_map_vit, current_count_vit, current_word_vit, word_rel_maps_vit,
                )
            (word_rel_maps_all_generated_token, current_rel_map_all_generated_token,
             current_count_all_generated_token, current_word_all_generated_token) = \
                compute_word_rel_map(
                    tokens, target_index, R_i_i_all_generated_token, SEPARATORS_LIST,
                    current_rel_map_all_generated_token, current_count_all_generated_token,
                    current_word_all_generated_token, word_rel_maps_all_generated_token,
                )

    # Store the last word's relevancy map.
    word_rel_maps_llama[current_word] = current_rel_map / current_count
    if enable_vit_relevancy:
        word_rel_maps_all[current_word_all] = current_rel_map_all / current_count_all
        word_rel_maps_vit[current_word_vit] = current_rel_map_vit / current_count_vit
        word_rel_maps_all_generated_token[current_word_all_generated_token] = (
            current_rel_map_all_generated_token / current_count_all_generated_token
        )

    return {
        "llama": word_rel_maps_llama,
        "llama_token": rel_maps_dict,
        "vit": word_rel_maps_vit,
        "all": word_rel_maps_all,
        "all_v2": word_rel_maps_all_generated_token,
    }
