import torch
from PIL import Image
import os.path as osp
import sys
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

# added by Yizhou
from typing import Any, Dict, List
from torchtune import config, training, utils
from torchtune.data import load_image, Message, padded_collate_tiled_images_and_mask
from omegaconf import DictConfig, OmegaConf
from torchtune.generation import sample

from torchtune.modules.transforms import Transform

import re


class SingleTurnYAMLToMessages(Transform):
    """
    Converts a single turn conversation in YAML format to a list of messages.

    Expects the YAML to look like:
        system: You are a helpful AI assistant.
        user: What is the capital of France?

    or if it includes an image:
        system: You are a helpful AI assistant.
        user:
            image: url or path_to_image
            text: Describe the image in detail.
    """

    def __call__(self, prompt: Dict[str, Any]) -> List[Message]:
        messages = []

        # Iterate through roles and add content
        for role, content in prompt.items():
            if isinstance(content, str):
                new_content = [{"type": "text", "content": content}]
            else:
                assert (
                    "image" in content.keys()
                ), "Multiple entries per role expect an image key"
                image_loc = content["image"]
                image = load_image(image_loc)
                new_content = [
                    {"type": "image", "content": image},
                    {"type": "text", "content": content["text"]},
                ]
            messages.append(Message(role=role, content=new_content))

        # Finally, add an empty assistant message to kick-start generation
        messages.append(Message(role="assistant", content=""))
        return messages


class llama_vision_efficient_lora_decode(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    # This function is used to split Llama-3.2-90B
    def split_model(self):
        import math
        device_map = {}
        num_gpus = torch.cuda.device_count()
        rank, world_size = get_rank_and_world_size()
        num_gpus = num_gpus // world_size

        num_layers = 100
        # GPU0: -5, GPU-1: -7
        total_cost = num_layers + 5 + 7

        # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
        num_layers_per_gpu = total_cost // num_gpus
        num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
        # The total number of GPUs might be odd
        num_layers_per_gpu[-1] = total_cost - sum(num_layers_per_gpu[:-1])
        num_layers_per_gpu[0] -= 5
        num_layers_per_gpu[-1] -= 7

        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
                layer_cnt += 1

        device_map['vision_model'] = rank
        device_map['language_model.model.embed_tokens'] = rank
        device_map['language_model.model.rotary_emb'] = rank
        device_map['language_model.model.norm'] = rank + world_size * (num_gpus - 1)
        device_map['language_model.lm_head'] = rank + world_size * (num_gpus - 1)
        device_map['multi_modal_projector'] = rank + world_size * (num_gpus - 1)
        return device_map

    def __init__(self, model_path='meta-llama/Llama-3.2-11B-Vision-Instruct', cfg = '/sensei-fs/users/yizhouw/projects/xuan/VLMEvalKit/configs/3-llama3_2_vision-11b-generation-lora-decode.yaml', **kwargs):
        try:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
        except Exception as e:
            logging.critical('Please install transformers>=4.45.0 before using llama_vision.')
            raise e

        self.cfg = OmegaConf.load(cfg)

        rank, world_size = get_rank_and_world_size()

        # TODO: 
        # 1. replace the model with the built with torchtune

        # assert world_size == 1, 'We only support world_size == 1 when AUTO_SPLIT is set for Llama-3.2-11B'
        # logging.warning('Currently, we only support to split the 11B model across all GPUs.')
        


        # Load checkpointer and state_dict
        _checkpointer = config.instantiate(self.cfg.checkpointer)
        _ckpt_dict = _checkpointer.load_checkpoint()

        self._device = utils.get_device(device=self.cfg.device)

        # Xuan: load lora weights
        if hasattr(self.cfg, "lora_adapter_path") and self.cfg.lora_adapter_path is not None:
            _lora_dict = torch.load(self.cfg.lora_adapter_path, map_location=self._device)
            _ckpt_dict[training.MODEL_KEY].update(_lora_dict)



        self._dtype = training.get_dtype(dtype=self.cfg.dtype, device=self._device)

        # Instantiate model
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(self.cfg.model)
        model.load_state_dict(_ckpt_dict[training.MODEL_KEY])
        self.model = model.cuda().eval()

        # insert from 4
        """Setup the model and transforms."""
        # Load checkpointer and state_dict
        _checkpointer_decoder = config.instantiate(self.cfg.checkpointer_decoder)
        _ckpt_dict_decoder_summary = _checkpointer_decoder.load_checkpoint()
        _ckpt_dict_decoder_caption = _checkpointer_decoder.load_checkpoint()
        _ckpt_dict_decoder_reasoning = _checkpointer_decoder.load_checkpoint()

        # Xuan: add lora and projector weights to state dict
        _ckpt_dict_decoder_summary = self.build_decoder(
            _ckpt_dict_decoder_summary, self.cfg.lora_adapter_path_decoder_summary, self.cfg.projector_weight_path_summary
        )
        _ckpt_dict_decoder_caption = self.build_decoder(
            _ckpt_dict_decoder_caption, self.cfg.lora_adapter_path_decoder_caption, self.cfg.projector_weight_path_caption
        )
        _ckpt_dict_decoder_reasoning = self.build_decoder(
            _ckpt_dict_decoder_reasoning, self.cfg.lora_adapter_path_decoder_reasoning, self.cfg.projector_weight_path_reasoning
        )

        # Instantiate model
        with training.set_default_dtype(self._dtype), self._device:
            model_decoder_summary = config.instantiate(self.cfg.model_decoder)
        model_decoder_summary.load_state_dict(_ckpt_dict_decoder_summary[training.MODEL_KEY])
        self.model_decoder_summary = model_decoder_summary
        with training.set_default_dtype(self._dtype), self._device:
            model_decoder_caption = config.instantiate(self.cfg.model_decoder)
        model_decoder_caption.load_state_dict(_ckpt_dict_decoder_caption[training.MODEL_KEY])
        self.model_decoder_caption = model_decoder_caption
        with training.set_default_dtype(self._dtype), self._device:
            model_decoder_reasoning = config.instantiate(self.cfg.model_decoder)
        model_decoder_reasoning.load_state_dict(_ckpt_dict_decoder_reasoning[training.MODEL_KEY])
        self.model_decoder_reasoning = model_decoder_reasoning
        # self._logger.info(f"Model decoders were initialized with precision {self._dtype}.")



        self.device = 'cuda'
        self.processor = AutoProcessor.from_pretrained(model_path)
        if 'Instruct' in model_path:
            kwargs_default = dict(do_sample=True, temperature=0.6, top_p=0.9)
        else:
            kwargs_default = dict(do_sample=False, max_new_tokens=512, temperature=0.0, top_p=None, num_beams=1)
        kwargs.update(kwargs_default)
        print(f'Following kwargs received: {kwargs}, will use as generation config. ')
        self.kwargs = kwargs
        self.model_name = model_path

        # Instantiate transforms
        self.model_transform = config.instantiate(self.cfg.tokenizer)
        self.to_messages = SingleTurnYAMLToMessages()

        self._token_idx_thinking_of_summary = self.model_transform.tokenizer.special_tokens['<THINKING_OF_SUMMARY>']     # 128013
        self._token_idx_thinking_of_caption = self.model_transform.tokenizer.special_tokens['<THINKING_OF_CAPTION>']     # 128014
        self._token_idx_thinking_of_reasoning = self.model_transform.tokenizer.special_tokens['<THINKING_OF_REASONING>'] # 128015

    def build_decoder(self, _ckpt_dict_decoder, lora_adapter_path_decoder, projector_weight_path):
        # Xuan: load lora weights
        _lora_dict = torch.load(lora_adapter_path_decoder, map_location=self._device)
        _ckpt_dict_decoder[training.MODEL_KEY].update(_lora_dict)
        
        # Xuan: load projector weights
        _projector_dict = torch.load(projector_weight_path, map_location=self._device)
        _ckpt_dict_decoder[training.MODEL_KEY].update(_projector_dict)

        return _ckpt_dict_decoder



    def use_custom_prompt(self, dataset):
        if dataset is None:
            return False
        if listinstr(['AI2D', 'MMMU', 'MathVista', 'ChartQA', 'DocVQA'], dataset):
            # For Certain dataset we use custom prompt
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        if listinstr(['AI2D'], dataset):
            self.kwargs['max_new_tokens'] = 400
            for key, item in options.items():
                question += f'\n{key}. {item}'
            if '11B' in self.model_name:
                prompt = (
                    f'Look at the scientific diagram carefully and answer the following question: {question}\n'
                    f'Think step by step and finally respond to the question '
                    f"with only the correct option number as \"FINAL ANSWER\"."
                    f"<cot_start>Let's think step by step."
                )
            elif '90B' in self.model_name:
                prompt = (
                    f'Look at the scientific diagram carefully and answer the following question: {question}\n'
                    f'Respond only with the correct option digit.'
                )
        elif listinstr(['MMMU'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            options = '\n'.join([f'{key}. {item}' for key, item in options.items()])
            prompt = (
                f'Look at the image carefully and solve the following question step-by-step. '
                f'Question: {question} Options: {options} Indicate the correct answer at the end.'
            )
            for i in range(len(tgt_path)):
                prompt = prompt.replace(f'<image {i+1}>', '')
        elif listinstr(['MathVista'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            prompt = f'{question}'
        elif listinstr(['ChartQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            if '11B' in self.model_name:
                prompt = (
                    f'You are provided a chart image and will be asked a question. '
                    f'You have to think through your answer and provide a step-by-step solution. '
                    f'Once you have the solution, write the final answer in at most a few words at the end '
                    f"with the phrase \"FINAL ANSWER:\". "
                    f"The question is: {question}<cot_start>Let's think step by step."
                )
            elif '90B' in self.model_name:
                prompt = (
                    f'You are provided a chart image and will be asked a question. '
                    f'Follow these steps carefully:\n '
                    f'Step 1: Analyze the question to understand what specific data or information is being asked for. '
                    f'Focus on whether the question is asking for a specific number or category '
                    f'from the chart image.\n '
                    f'Step 2: Identify any numbers, categories, or groups mentioned in the question '
                    f'and take note of them. Focus on detecting and matching them directly to the image. \n'
                    f'Step 3: Study the image carefully and find the relevant data corresponding to the categories '
                    f'or numbers mentioned. Avoid unnecessary assumptions or calculations; '
                    f'simply read the correct data from the image.\n '
                    f'Step 4: Develop a clear plan to solve the question by locating the right data. '
                    f'Focus only on the specific category or group that matches the question. \n'
                    f'Step 5: Use step-by-step reasoning to ensure you are referencing the correct numbers '
                    f'or data points from the image, avoiding unnecessary extra steps or interpretations.\n '
                    f"Step 6: Provide the final answer, starting with \"FINAL ANSWER:\" "
                    f'and using as few words as possible, '
                    f'simply stating the number or data point requested. \n\n '
                    f"The question is: {question}<cot_start>Let's think step by step."
                )
        elif listinstr(['DocVQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            prompt = (
                f'Read the text in the image carefully and answer the question '
                f'with the text as seen exactly in the image. '
                f'For yes/no questions, just respond Yes or No. '
                f'If the answer is numeric, just respond with the number and nothing else. '
                f'If the answer has multiple words, just respond with the words and absolutely nothing else. '
                f'Never respond in a sentence or a phrase.\n Question: {question}'
            )
        else:
            raise NotImplementedError(f'Dataset {dataset}) not supported.')

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    @torch.inference_mode()
    def generate_detail(self, current_prompt, cfg: DictConfig):
        # 1. Convert input to messages
        messages = self.to_messages(OmegaConf.to_container(current_prompt))
        is_multimodal_input = any([m.contains_media for m in messages])

        # 2. Apply model transform
        model_inputs = self.model_transform({"messages": messages}, inference=True)
        seq_len = len(model_inputs["tokens"])
        total_response_length = seq_len + self.cfg.max_new_tokens
        
        # 3. Setup KV cache
        with self._device:
            self.model.setup_caches(
                batch_size=1,
                dtype=self._dtype,
                encoder_max_seq_len=(
                    self.model_transform.image_seq_len if is_multimodal_input else None
                ),
                decoder_max_seq_len=total_response_length,
            )

        # 4. Pre-allocate causal mask and input_pos
        causal_mask = torch.tril(
            torch.ones(
                size=(total_response_length, total_response_length),
                dtype=torch.bool,
                device=self._device,
            )
        )
        input_pos = torch.arange(total_response_length)

        # 5. Collate to batch size of 1 and tensor-ify
        batch = {}
        if is_multimodal_input:
            batch = padded_collate_tiled_images_and_mask(
                [model_inputs],
                pad_direction="left",
                pad_max_images=1,
                pad_max_tiles=self.model_transform.max_num_tiles,
            )
            batch["encoder_mask"] = batch["encoder_mask"][:, :seq_len]
            prompt = batch.pop("tokens").to(self._device)
        else:
            prompt = torch.tensor(
                model_inputs["tokens"], device=self._device
            ).unsqueeze(0)
        batch["mask"] = causal_mask[None, :seq_len]
        batch["input_pos"] = input_pos[None, :seq_len]
        utils.batch_to_device(batch, self._device)

        # 6. Prefill step
        generated_tokens = []
        last_hidden_state_records = []  # Xuan: record last hidden state
        # t0 = time.perf_counter()
        logits = self.model(prompt, **batch)[:, -1]
        token = sample(logits, temperature=self.cfg.temperature, top_k=self.cfg.top_k)
        generated_tokens.append(token.item())
        last_hidden_state_records.append(self.model.decoder.last_hidden_state)

        if is_multimodal_input:
            # Don't need image info b/c we only support 1 image and it's been
            # processed by the model now
            batch.pop("encoder_input")
            batch["encoder_mask"] = batch["encoder_mask"][:, -1:]

        # 7. Continue generating
        for i in range(self.cfg.max_new_tokens):

            # Update position and mask for incremental decoding
            batch["input_pos"] = input_pos[None, seq_len]
            batch["mask"] = causal_mask[None, seq_len, None, :]

            if token.item() in self.model_transform.stop_tokens:
                break

            logits = self.model(token, **batch)[:, -1]
            token = sample(logits, temperature=self.cfg.temperature, top_k=self.cfg.top_k)
            generated_tokens.append(token.item())
            seq_len += 1

            last_hidden_state_records.append(self.model.decoder.last_hidden_state)

        # t = time.perf_counter() - t0

        # 8. Translate tokens back to text
        generated_tokens_for_translate = [e for e in generated_tokens if e < 128255]
        decoded = self.model_transform.decode(generated_tokens_for_translate, skip_special_tokens=True)

        return decoded, generated_tokens, last_hidden_state_records

    def get_middle_sublist(self, input_list, start_sublist, end_sublist):
        # Find the indices of the start and end sublists
        try:
            start_index = next(i for i in range(len(input_list)) if input_list[i:i+len(start_sublist)] == start_sublist)
            end_index = next(i for i in range(start_index + len(start_sublist), len(input_list)) if input_list[i:i+len(end_sublist)] == end_sublist)
            
            # Indices for the middle sublist
            middle_start_index = start_index + len(start_sublist)
            middle_end_index = end_index
            
            return middle_start_index, middle_end_index
        except StopIteration:
            return None, None

    @torch.inference_mode()
    def generate_detail_decoder(self, current_prompt, cfg: DictConfig, generated_token_idx, last_hidden_state_records, stage):
        # 0. Xuan: remove image info in the prompt
        current_prompt = current_prompt.copy()  # Xuan: avoid changing the original prompt
        if "image" in current_prompt.user:  # Xuan: pure LLM decoder does not need image info
            current_prompt.user.pop("image")
        if "<image>" in current_prompt.user.text:
            current_prompt.user = current_prompt.user.text.replace("<image>", "")
        else:
            current_prompt.user = current_prompt.user.text  # Xuan: avoid assert at line 51
        if stage == 'summary':
            current_model_decoder = self.model_decoder_summary
        elif stage == 'caption':
            current_model_decoder = self.model_decoder_caption
        elif stage == 'reasoning':
            current_model_decoder = self.model_decoder_reasoning

        # 1. Convert input to messages
        messages = self.to_messages(OmegaConf.to_container(current_prompt))
        is_multimodal_input = False

        # 2. Apply model transform
        model_inputs = self.model_transform({"messages": messages}, inference=True)
        seq_len = len(model_inputs["tokens"])
        total_response_length = seq_len + self.cfg.max_new_tokens
        
        # Xuan: get mask_special_token according to the stage
        if stage == 'summary':
            mask_special_token = torch.from_numpy(np.array(model_inputs['tokens']) == self._token_idx_thinking_of_summary).unsqueeze(0)
        elif stage == 'caption':
            mask_special_token = torch.from_numpy(np.array(model_inputs['tokens']) == self._token_idx_thinking_of_caption).unsqueeze(0)
        elif stage == 'reasoning':
            mask_special_token = torch.from_numpy(np.array(model_inputs['tokens']) == self._token_idx_thinking_of_reasoning).unsqueeze(0)
        
        # Xuan: find special token in the generated_token_idx
            # [19389, 2864, 49970, 29, 220] -> '<SUMMARY> '
            # [694, 28477, 49970, 1363] -> '</SUMMARY>\n\n'
            # [20996, 2599, 60459, 29, 220] -> '<CAPTION> '
            # [694, 32500, 60459, 1363] -> '</CAPTION>\n\n'
            # [27, 793, 36404, 1753, 29, 220] -> '<REASONING> '
            # [694, 793, 36404, 1753, 1363] -> '</REASONING>\n\n'
        # Therefore, in the generated_token_idx, we can find the corresponding token:
            # [19389, 2864, 49970, 29, 220, 
            # 128xxx, 
            # 694, 28477, 49970, 1363, 
            # 20996, 2599, 60459, 29, 220, 
            # 128xxx, 
            # 694, 32500, 60459, 1363, 
            # 27, 793, 36404, 1753, 29, 220, 
            # 128xxx, 
            # 694, 793, 36404, 1753, 1363, 
            # ...]
        if stage == 'summary':
            thinking_token_idx_start, thinking_token_idx_end = self.get_middle_sublist(generated_token_idx, [19389, 2864, 49970, 29, 220], [694, 28477, 49970, 1363])
            if thinking_token_idx_start is None:
                return None
            thinking_token_main_pipeline = torch.cat(last_hidden_state_records[thinking_token_idx_start : thinking_token_idx_end], dim=1)   # shape: [B, N, D]
            assert thinking_token_main_pipeline.shape[1] == thinking_token_idx_end - thinking_token_idx_start, "The shape of thinking_token_main_pipeline is not correct"
        elif stage == 'caption':
            thinking_token_idx_start, thinking_token_idx_end = self.get_middle_sublist(generated_token_idx, [20996, 2599, 60459, 29, 220], [694, 32500, 60459, 1363])
            if thinking_token_idx_start is None:
                return None
            thinking_token_main_pipeline = torch.cat(last_hidden_state_records[thinking_token_idx_start : thinking_token_idx_end], dim=1)   # shape: [B, N, D]
            assert thinking_token_main_pipeline.shape[1] == thinking_token_idx_end - thinking_token_idx_start, "The shape of thinking_token_main_pipeline is not correct"
        elif stage == 'reasoning':
            thinking_token_idx_start, thinking_token_idx_end = self.get_middle_sublist(generated_token_idx, [27, 793, 36404, 1753, 29, 220], [694, 793, 36404, 1753, 1363])
            if thinking_token_idx_start is None:
                return None
            thinking_token_main_pipeline = torch.cat(last_hidden_state_records[thinking_token_idx_start : thinking_token_idx_end], dim=1)   # shape: [B, N, D]
            assert thinking_token_main_pipeline.shape[1] == thinking_token_idx_end - thinking_token_idx_start, "The shape of thinking_token_main_pipeline is not correct"

        # 3. Setup KV cache
        with self._device:
            current_model_decoder.setup_caches(
                batch_size=1,
                dtype=self._dtype,
                encoder_max_seq_len=(
                    self.model_transform.image_seq_len if is_multimodal_input else None
                ),
                decoder_max_seq_len=total_response_length,
            )

        # 4. Pre-allocate causal mask and input_pos
        causal_mask = torch.tril(
            torch.ones(
                size=(total_response_length, total_response_length),
                dtype=torch.bool,
                device=self._device,
            )
        )
        input_pos = torch.arange(total_response_length)

        # 5. Collate to batch size of 1 and tensor-ify
        batch = {}
        if is_multimodal_input:
            batch = padded_collate_tiled_images_and_mask(
                [model_inputs],
                pad_direction="left",
                pad_max_images=1,
                pad_max_tiles=self.model_transform.max_num_tiles,
            )
            batch["encoder_mask"] = batch["encoder_mask"][:, :seq_len]
            prompt = batch.pop("tokens").to(self._device)
        else:
            prompt = torch.tensor(
                model_inputs["tokens"], device=self._device
            ).unsqueeze(0)
        batch["mask"] = causal_mask[None, :seq_len]
        batch["input_pos"] = input_pos[None, :seq_len]
        utils.batch_to_device(batch, self._device)

        # 6. Prefill step
        generated_tokens = []
        # t0 = time.perf_counter()
        logits = current_model_decoder(prompt, **batch, 
                                    thinking_token=thinking_token_main_pipeline,
                                    thinking_token_mask=mask_special_token.to(self._device),
                                    )[:, -1]
        token = sample(logits, temperature=self.cfg.temperature, top_k=self.cfg.top_k)
        generated_tokens.append(token.item())

        if is_multimodal_input:
            # Don't need image info b/c we only support 1 image and it's been
            # processed by the model now
            batch.pop("encoder_input")
            batch["encoder_mask"] = batch["encoder_mask"][:, -1:]

        # 7. Continue generating
        for i in range(self.cfg.max_new_tokens):

            # Update position and mask for incremental decoding
            batch["input_pos"] = input_pos[None, seq_len]
            batch["mask"] = causal_mask[None, seq_len, None, :]

            if token.item() in self.model_transform.stop_tokens:
                break

            logits = current_model_decoder(token, **batch)[:, -1]
            token = sample(logits, temperature=self.cfg.temperature, top_k=self.cfg.top_k)
            generated_tokens.append(token.item())
            seq_len += 1

        # t = time.perf_counter() - t0

        # 8. Translate tokens back to text
        generated_tokens = [e for e in generated_tokens if e < 128255]
        decoded = self.model_transform.decode(generated_tokens, skip_special_tokens=True)

        return decoded
    
    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        # import pdb; pdb.set_trace()

        # image = Image.open(image_path)
        # messages = [
        #     {'role': 'user', 'content': [
        #         {'type': 'image'},
        #         {'type': 'text', 'text': prompt}
        #     ]}
        # ]
        current_prompt = OmegaConf.create({
                # "system": "You are a helpful AI assistant.",
                "user": {
                    "image": image_path,
                    "text": prompt
                }
            })
        summary_prompt = "For the question: '{}' Can you explain the thinking progress "+"<THINKING_OF_SUMMARY>"*self.cfg.num_thinking_of_summary+"?".format(prompt)
        current_prompt_summary = OmegaConf.create({
                # "system": "You are a helpful AI assistant.",
                "user": {
                    "image": image_path,
                    "text": summary_prompt
                }
            })
        caption_prompt = "For the question: '{}' Can you explain the thinking progress "+"<THINKING_OF_CAPTION>"*self.cfg.num_thinking_of_caption+"?".format(prompt)
        current_prompt_caption = OmegaConf.create({
                # "system": "You are a helpful AI assistant.",
                "user": {
                    "image": image_path,
                    "text": caption_prompt
                }
            })
        reasoning_prompt = "For the question: '{}' Can you explain the thinking progress "+"<THINKING_OF_REASONING>"*self.cfg.num_thinking_of_reasoning+"?".format(prompt)
        current_prompt_reasoning = OmegaConf.create({
            # "system": "You are a helpful AI assistant.",
            "user": {
                "image": image_path,
                "text": reasoning_prompt
            }
        })
        # current_question_caption = current_sample['conversations_caption'][0]['value']
        # # current_answer_caption = current_sample['conversations_caption'][1]['value']
        # current_prompt_caption = OmegaConf.create({
        #     # "system": "You are a helpful AI assistant.",
        #     "user": {
        #         "image": image_path,
        #         "text": current_question_caption
        #     }
        # })
        # current_question_reasoning = current_sample['conversations_reasoning'][0]['value']
        # # current_answer_reasoning = current_sample['conversations_reasoning'][1]['value']
        # current_prompt_reasoning = OmegaConf.create({
        #     # "system": "You are a helpful AI assistant.",
        #     "user": {
        #         "image": image_path,
        #         "text": current_question_reasoning
        #     }
        # })

        decoded, generated_token_idx, last_hidden_state_records = self.generate_detail(current_prompt, self.cfg)
        decoded_text_summary = self.generate_detail_decoder(current_prompt_summary, self.cfg, generated_token_idx, last_hidden_state_records, 'summary')
        print("decoded_text_summary: {}".format(decoded_text_summary))
        decoded_text_caption = self.generate_detail_decoder(current_prompt_caption, self.cfg, generated_token_idx, last_hidden_state_records, 'caption')
        print("decoded_text_caption: {}".format(decoded_text_caption))
        decoded_text_reasoning = self.generate_detail_decoder(current_prompt_reasoning, self.cfg, generated_token_idx, last_hidden_state_records, 'reasoning')
        print("decoded_text_reasoning: {}".format(decoded_text_reasoning))

        # # step 1 -> step 8
        # # 1. Convert input to messages
        # messages = self.to_messages(OmegaConf.to_container(current_prompt))
        # is_multimodal_input = any([m.contains_media for m in messages])

        # # 2. Apply model transform
        # model_inputs = self.model_transform({"messages": messages}, inference=True)
        # seq_len = len(model_inputs["tokens"])
        # total_response_length = seq_len + self.cfg.max_new_tokens

        # # 3. Setup KV cache
        # with self._device:
        #     self.model.setup_caches(
        #         batch_size=1,
        #         dtype=self._dtype,
        #         encoder_max_seq_len=(
        #             self.model_transform.image_seq_len if is_multimodal_input else None
        #         ),
        #         decoder_max_seq_len=total_response_length,
        #     )

        # # 4. Pre-allocate causal mask and input_pos
        # causal_mask = torch.tril(
        #     torch.ones(
        #         size=(total_response_length, total_response_length),
        #         dtype=torch.bool,
        #         device=self._device,
        #     )
        # )
        # input_pos = torch.arange(total_response_length)

        # # 5. Collate to batch size of 1 and tensor-ify
        # batch = {}
        # if is_multimodal_input:
        #     batch = padded_collate_tiled_images_and_mask(
        #         [model_inputs],
        #         pad_direction="left",
        #         pad_max_images=1,
        #         pad_max_tiles=self.model_transform.max_num_tiles,
        #     )
        #     batch["encoder_mask"] = batch["encoder_mask"][:, :seq_len]
        #     prompt = batch.pop("tokens").to(self._device)
        # else:
        #     prompt = torch.tensor(
        #         model_inputs["tokens"], device=self._device
        #     ).unsqueeze(0)
        # batch["mask"] = causal_mask[None, :seq_len]
        # batch["input_pos"] = input_pos[None, :seq_len]
        # utils.batch_to_device(batch, self._device)

        # # 6. Prefill step
        # generated_tokens = []
        # # t0 = time.perf_counter()
        # logits = self.model(prompt, **batch)[:, -1]
        # token = sample(logits, temperature=self.cfg.temperature, top_k=self.cfg.top_k)
        # generated_tokens.append(token.item())

        # if is_multimodal_input:
        #     # Don't need image info b/c we only support 1 image and it's been
        #     # processed by the model now
        #     batch.pop("encoder_input")
        #     batch["encoder_mask"] = batch["encoder_mask"][:, -1:]

        # # 7. Continue generating
        # for i in range(self.cfg.max_new_tokens):

        #     # Update position and mask for incremental decoding
        #     batch["input_pos"] = input_pos[None, seq_len]
        #     batch["mask"] = causal_mask[None, seq_len, None, :]

        #     if token.item() in self.model_transform.stop_tokens:
        #         break

        #     logits = self.model(token, **batch)[:, -1]
        #     token = sample(logits, temperature=self.cfg.temperature, top_k=self.cfg.top_k)
        #     generated_tokens.append(token.item())
        #     seq_len += 1

        # # t = time.perf_counter() - t0

        # # 8. Translate tokens back to text
        # # decoded = self.model_transform.decode(generated_tokens, skip_special_tokens=True)

        # generated_tokens_for_translate = [e for e in generated_tokens if e < 128255]
        # decoded = self.model_transform.decode(generated_tokens_for_translate, skip_special_tokens=True)

        # Extract the conclusion part while preserving the tags
        if match := re.search(r'(<CONCLUSION>.*?</CONCLUSION>)', decoded, re.DOTALL):
            decoded = match.group(1)

        print(decoded)
        import pdb; pdb.set_trace()
        return decoded


        # input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        # inputs = self.processor(image, input_text, return_tensors='pt').to(self.device)
        # if not self.use_custom_prompt(dataset):
        #     if dataset is not None and DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
        #         self.kwargs['max_new_tokens'] = 128
        #     else:
        #         self.kwargs['max_new_tokens'] = 512
        # output = self.model.generate(**inputs, **self.kwargs)
        # return self.processor.decode(output[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '')
