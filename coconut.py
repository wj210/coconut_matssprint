# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
Outputs_latent = namedtuple("Outputs_latent", ["loss", "inputs_embeds", "logits","latents"])
MAX_N_LATENT = 8


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids,store_latent = False,insert_latent=None,insert_step = -1,**kwargs):
        """
        store_latent outputs the latent token states
        insert_latent is the token to be inserted (intervene)
        insert_step is the step to insert the token
        """

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        stored_latents = [] # to store latents

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            sample_stored_latents = []
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                if insert_latent is not None and pass_idx == insert_step:
                    tensor_list[batch_idx][token_idx] = insert_latent[pass_idx].to(
                        inputs_embeds.device
                    ) # insert latent is n_pass, dim
                else:
                    # replace it with the preceding last hidden states
                    tensor_list[batch_idx][token_idx] = hidden_states[
                        batch_idx, token_idx - 1 - hidden_states_offset, :
                    ]
                if store_latent:
                    sample_stored_latents.append(hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ])
            if store_latent:
                stored_latents.append(
                    torch.stack(sample_stored_latents, dim=0).unsqueeze(1)) # B, pass, dim

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )
        
        if store_latent:
            stored_latents = torch.cat(stored_latents, dim=1) # B, n_pass, dim

        # final pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
            # use_cache = True if past_kv else False,
        )

        logits.append(outputs.logits)
        logits = torch.cat(logits, dim=-2)
        self.gen_forward_cnt += max_n_latents + 1
        if not store_latent:
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)
        else:
            return Outputs_latent(
                loss=None,
                inputs_embeds=inputs_embeds,
                logits=logits,
                latents=stored_latents,
            )

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        return_logits = False, # return output logits
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device), 
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
            **kwargs
        )
        inputs_embeds = outputs.inputs_embeds
        output_logits = []

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)

        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)

        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            if return_logits:
                output_logits.append(outputs.logits[0,-1].detach().cpu())
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)


        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        elif return_logits:
            return torch.tensor(tokens).view(1, -1), torch.stack(output_logits, dim=0)
        else:
            return torch.tensor(tokens).view(1, -1)


    def batch_generate(
        self,
        input_ids,
        attention_mask,
        position_ids,
        max_new_tokens=16,
    ):

        self.gen_forward_cnt = 0
        output_tokens = []
        labels = input_ids.clone()  # placeholder. not used.
        bz = input_ids.shape[0]
        outputs = self.forward(
            input_ids,
            attention_mask, # past actual attention_mask
            labels,
            position_ids,
            past_kv=True,
        )

        inputs_embeds = outputs.inputs_embeds
        next_token = torch.argmax(outputs.logits[:, -1],dim = -1)
        output_tokens.append(next_token)
        new_token_embed = self.embedding(next_token).view(bz, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)
        finished_seq = torch.zeros(bz).to(input_ids.device).bool()
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds[:,-1:],past_key_values=outputs.past_key_values,use_cache=True)
            next_token = torch.argmax(outputs.logits[:, -1],dim = -1)
            # check for each batch if next token is eos, if yes, update finished_seq
            finished_seq = finished_seq | (next_token == self.eos_token_id)
            output_tokens.append(next_token)
            if finished_seq.all():
                break
            new_token_embed = self.embedding(next_token).view(bz, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)
        output_tokens = torch.stack(output_tokens, dim=1)
        return output_tokens


            

