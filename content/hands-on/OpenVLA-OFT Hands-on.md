---
title: OpenVLA-OFT是如何将VLM改造成VLA的
draft: false
tags:
  - VLA
  - Hands-on
date: 2025-10-16
---
# 前言

在[[OpenVLA Hands-on]]中已经介绍了OpenVLA的改造。[[OpenVLA-OFT]]我们也介绍过，是在[[OpenVLA]]后对于VLA范式的探索，具体内容就不展开叙述了。OpenVLA-OFT仓库是OpenVLA仓库的一个fork，在本文中研究OpenVLA-OFT对于VLM的改造。

![[file-20251020141718426.png]]

# 环境配置

原始仓库地址： https://github.com/moojink/openvla-oft

相比于OpenVLA每个LIBERO任务都有一个模型，这次提供了一个all-in-one的模型 https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 。

首先根据`SETUP.md`安装对应的环境，再根据`LIBERO.md`安装对应的LIBERO环境。之后运行

```bash
pip install numpy==1.26.0
pip install opencv-python==4.6.0.66
```

就可以把环境配置完成了。具体评测运行同样可以参考`LIBERO.md`中的指令。

# 代码分析

## 原始VLM改造

OpenVLA对VLM的forward函数是没有修改的，在OpenVLA-OFT中，由于加入了本体状态和任务描述的FiLM，对forward函数进行了改动：

```python title="prismatic/extern/hf/modeling_prismatic.py#PrismaticForConditionalGeneration"
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_projector_features: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    proprio=None,
    proprio_projector=None,
    noisy_actions=None,
    noisy_action_projector=None,
    diffusion_timestep_embeddings=None,
    use_film: bool = False,
) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
    """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    output_projector_features = output_projector_features if output_projector_features is not None else False
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
    use_cache = use_cache and not self.training

    # Instantiate Placeholder for Projector Features
    projected_patch_embeddings = None

    # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
    if input_ids.shape[1] == 1:
        assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
        assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
        assert labels is None, "Unexpected key `labels` provided during cached generation!"

        language_model_output = self.language_model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=None,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # === Handle Unimodal Forward ===
    elif pixel_values is None:
        assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
        assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

        language_model_output = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # === Handle Multimodal Forward ===
    elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
        assert past_key_values is None, "Unexpected key `past_key_values` provided during multimodal forward!"

        # Get input embeddings (from language model embeddings)
        input_embeddings = self.get_input_embeddings()(input_ids)  # (B, seq_len, D)

        # Extract action masks
        all_actions_mask = self._process_action_masks(labels)

        # Extract the language portion of the input embeddings (i.e. remove the action tokens portion)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )  # (B, lang_seq_len, llm_dim)

        # Get visual features
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        # Add proprioceptive state if provided
        projected_patch_embeddings = self._process_proprio_features(
            projected_patch_embeddings, proprio, proprio_projector
        )

        # [Diffusion] Add diffusion timestep embedding if provided
        if diffusion_timestep_embeddings is not None:
            # For simplicity, just append diffusion timestep embedding to the end of projected vision patch tokens
            projected_patch_embeddings = torch.cat(
                (projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
            )

        # Process action embeddings
        if noisy_actions is not None:
            # Get mask corresponding to all action tokens
            all_actions_mask = self._process_action_masks(labels)

            # Reshape noisy actions into individual action tokens
            # noisy_actions: (B, chunk_len, action_dim) -> (B, chunk_len * action_dim, 1)
            B = noisy_actions.shape[0]
            noisy_actions = noisy_actions.reshape(B, -1).unsqueeze(-1)

            # Project noisy action tokens into language model embedding space
            noisy_action_features = noisy_action_projector(noisy_actions)  # (B, chunk_len * action_dim, llm_dim)

            # Replace embeddings of the action tokens with noisy action embeddings
            input_embeddings = self._replace_input_embeddings(
                input_embeddings, all_actions_mask, noisy_action_features
            )
        else:
            # Replace the embeddings of the action tokens with zeros
            # (Later on, the positional embeddings will be added to them)
            all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
            input_embeddings = input_embeddings * ~all_actions_mask

        # Build multimodal embeddings & attention mask
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Build labels for multimodal sequence if needed
        multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings)

        # Dispatch to language model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=multimodal_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # === Otherwise =>> Assume Invalid! ===
    elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
        raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

    else:
        raise ValueError(
            "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
            f"=> `input_ids` = {input_ids is not None}\n"
            f"=> `attention_mask` = {attention_mask is not None}\n"
            f"=> `pixel_values` = {pixel_values is not None}\n"
            f"=> `labels` = {labels is not None}\n"
            f"=> `input_embeds` = {inputs_embeds is not None}\n"
            f"=> `past_key_values` = {past_key_values is not None}\n"
            f"=> `use_cache` = {use_cache}"
        )

    # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
    if not return_dict:
        if output_projector_features and (projected_patch_embeddings is not None):
            return *language_model_output, projected_patch_embeddings

        return language_model_output

    return PrismaticCausalLMOutputWithPast(
        loss=language_model_output.loss,
        logits=language_model_output.logits,
        past_key_values=language_model_output.past_key_values,
        hidden_states=language_model_output.hidden_states,
        attentions=language_model_output.attentions,
        projector_features=projected_patch_embeddings,
    )
```

具体来讲，在输入参数里多了

```python title="prismatic/extern/hf/modeling_prismatic.py#PrismaticForConditionalGeneration"
proprio=None,
proprio_projector=None,
noisy_actions=None,
noisy_action_projector=None,
diffusion_timestep_embeddings=None,
use_film: bool = False,
```

在计算视觉嵌入时，判断`use_film`的值：

```python title="prismatic/extern/hf/modeling_prismatic.py#PrismaticForConditionalGeneration"
def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
    """Process vision features with optional FiLM conditioning"""
    if use_film:
        # FiLM: Infuse language inputs into visual features
        patch_features = self.vision_backbone(pixel_values, language_embeddings)  # (bsz, 256 * num_images, D)
    else:
        patch_features = self.vision_backbone(pixel_values)  # (bsz, 256 * num_images, D)

    # Project patch embeddings into language embedding space
    return self.projector(patch_features)
```

计算本体特征：

```python title="prismatic/extern/hf/modeling_prismatic.py#PrismaticForConditionalGeneration"
def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
    """Process proprioceptive features and append to vision features"""
    if proprio_projector is not None and proprio is not None:
        # projected_patch_embeddings: (bsz, num_patches * num_images, llm_dim)
        # proprio: (bsz, proprio_dim) or (propro_dim,)
        proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
        # For simplicity, just append proprio token to the end of projected vision patch tokens
        return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
    return projected_patch_embeddings
```

计算输入嵌入：

```python title="prismatic/extern/hf/modeling_prismatic.py#PrismaticForConditionalGeneration"
# [Diffusion] Add diffusion timestep embedding if provided
if diffusion_timestep_embeddings is not None:
    # For simplicity, just append diffusion timestep embedding to the end of projected vision patch tokens
    projected_patch_embeddings = torch.cat(
        (projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
    )

# Process action embeddings
if noisy_actions is not None:
    # Get mask corresponding to all action tokens
    all_actions_mask = self._process_action_masks(labels)

    # Reshape noisy actions into individual action tokens
    # noisy_actions: (B, chunk_len, action_dim) -> (B, chunk_len * action_dim, 1)
    B = noisy_actions.shape[0]
    noisy_actions = noisy_actions.reshape(B, -1).unsqueeze(-1)

    # Project noisy action tokens into language model embedding space
    noisy_action_features = noisy_action_projector(noisy_actions)  # (B, chunk_len * action_dim, llm_dim)

    # Replace embeddings of the action tokens with noisy action embeddings
    input_embeddings = self._replace_input_embeddings(
        input_embeddings, all_actions_mask, noisy_action_features
    )
else:
    # Replace the embeddings of the action tokens with zeros
    # (Later on, the positional embeddings will be added to them)
    all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
    input_embeddings = input_embeddings * ~all_actions_mask
```

all_action_masks用于判断是否为动作token，是则为1。

- Diffusion方式：在本体嵌入里额外加上Diffusion的时间步，并且利用noisy_action_projector来对noisy_actions进行去噪。如果是动作token则替换为加噪动作特征。
    
- 其他方式：如果是动作token则替换为全0向量。
    

## OpenVLAForActionPrediction

继承`PrismaticForConditionalGeneration`，是动作生成的核心模块。在OpenVLA原始实现中只是将VLM生成的token转为动作区间的离散index，再映射到连续动作空间。

![[file-20251020143924615.png]]

而在OpenVLA-OFT中，根据论文定义，考虑了三种生成方式，从而定义了三种不同的动作生成解码方法：

```python title="prismatic/extern/hf/modeling_prismatic.py#OpenVLAForActionPrediction"
def predict_action(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    unnorm_key: Optional[str] = None,
    proprio=None,
    proprio_projector=None,
    action_head=None,
    noisy_action_projector=None,
    use_film: bool = False,
    **kwargs: str,
) -> np.ndarray:
    """Predict actions from input sequence, with options for different prediction methods.

    Args:
        input_ids: Input token ids
        unnorm_key: Key for unnormalization statistics
        proprio: Proprioceptive features
        proprio_projector: Projector for proprioceptive features
        action_head: Optional head for L1 regression or diffusion-based prediction
        noisy_action_projector: Projector for noisy actions in diffusion-based prediction
        use_film: Whether to use FiLM conditioning
        **kwargs: Additional arguments including pixel_values and attention_mask

    Returns:
        Tuple of (unnormalized_actions, action_hidden_states)
    """
    # If the special empty token ('') does not already appear after the colon (':') token in the prompt
    # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        )

    pixel_values = kwargs["pixel_values"]
    attention_mask = kwargs["attention_mask"]

    # Create fake labels tensor (needed for action mask)
    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX

    # Get number of tokens in prompt (excluding the start token)
    NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token

    # Prepare inputs by adding necessary tokens
    input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

    # Update labels tensor for action mask computation later
    labels = self._prepare_labels_for_action_prediction(labels, input_ids)

    # Get input embeddings and action masks
    input_embeddings = self.get_input_embeddings()(input_ids)
    all_actions_mask = self._process_action_masks(labels)

    # Extract language embeddings
    language_embeddings = input_embeddings[~all_actions_mask].reshape(
        input_embeddings.shape[0], -1, input_embeddings.shape[2]
    )

    # Process vision features
    projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

    # Add proprioceptive features if provided
    use_proprio = proprio_projector is not None and proprio is not None
    if use_proprio:
        proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
        projected_patch_embeddings = self._process_proprio_features(
            projected_patch_embeddings, proprio, proprio_projector
        )

    # Use diffusion if provided, otherwise use regression or discrete prediction
    use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

    # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
    NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
    if use_proprio:
        NUM_PATCHES += 1
    if use_diffusion:
        NUM_PATCHES += 1

    if use_diffusion:
        # Sample random noise with shape equal to output action, used as the starting state for reverse diffusion
        noise = torch.randn(
            size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
        )

        # Run diffusion-based prediction
        normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
            input_embeddings,
            all_actions_mask,
            noise,
            action_head,
            projected_patch_embeddings,
            labels,
            attention_mask,
            NUM_PATCHES,
            NUM_PROMPT_TOKENS,
            noisy_action_projector,
        )
    else:
        # Run regression or discrete token-based prediction
        normalized_actions, actions_hidden_states = self._regression_or_discrete_prediction(
            input_embeddings,
            all_actions_mask,
            projected_patch_embeddings,
            attention_mask,
            labels,
            NUM_PATCHES,
            NUM_PROMPT_TOKENS,
            action_head,
        )

    # Unnormalize predicted actions
    actions = self._unnormalize_actions(normalized_actions, unnorm_key)

    return actions, actions_hidden_states
```

其中会根据不同的条件进入不同的动作头。

### 连续动作-Diffusion


```python title="prismatic/extern/hf/modeling_prismatic.py#OpenVLAForActionPrediction"
def _run_diffusion_prediction(
    self,
    input_embeddings,
    all_actions_mask,
    noise,
    action_head,
    projected_patch_embeddings,
    labels,
    attention_mask,
    NUM_PATCHES,
    NUM_PROMPT_TOKENS,
    noisy_action_projector,
):
    """Run diffusion-based action prediction"""
    # Clone embedding for reuse in each timestep
    orig_projected_patch_embeddings = projected_patch_embeddings.clone()
    curr_noisy_actions = noise

    # Reverse diffusion: Iteratively denoise to generate action prediction
    for t in action_head.noise_scheduler.timesteps:
        # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action
        # embedding, and diffusion timestep embedding)
        timesteps = torch.Tensor([t]).to(labels.device)
        diffusion_timestep_embeddings = (
            action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        # [Diffusion] Replace the embeddings of the action tokens with noisy actions
        # (Later on, the positional embeddings will be added to them)

        # For simplicity, append diffusion timestep embedding to the end of projected vision tokens
        projected_patch_embeddings = torch.cat(
            (orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
        )

        # Reshape and project noisy actions into language embedding space
        B = curr_noisy_actions.shape[0]
        orig_curr_noisy_actions_shape = curr_noisy_actions.shape
        curr_noisy_actions = curr_noisy_actions.reshape(B, -1).unsqueeze(-1)
        noisy_action_features = noisy_action_projector(curr_noisy_actions)
        curr_noisy_actions = curr_noisy_actions.reshape(orig_curr_noisy_actions_shape)

        # Replace action token embeddings with noisy action embeddings
        input_embeddings = self._replace_input_embeddings(
            input_embeddings.clone(), all_actions_mask, noisy_action_features
        )

        # Build multimodal embeddings and attention mask
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Forward pass through language model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action portion of response
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        actions_hidden_states = last_hidden_states[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            :,
        ]  # (B, act_chunk_len, D)

        # Predict noise and update noisy actions: x_t -> x_{t-1}
        noise_pred = action_head.predict_noise(actions_hidden_states)
        curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

    # Return final actions
    return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states
```

### 连续动作-回归和离散

```python title="prismatic/extern/hf/modeling_prismatic.py#OpenVLAForActionPrediction"
def _regression_or_discrete_prediction(
    self,
    input_embeddings,
    all_actions_mask,
    projected_patch_embeddings,
    attention_mask,
    labels,
    NUM_PATCHES,
    NUM_PROMPT_TOKENS,
    action_head=None,
):
    """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""
    # Zero out action token embeddings
    all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
    input_embeddings = input_embeddings * ~all_actions_mask

    # Build multimodal embeddings and attention mask
    multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
        input_embeddings, projected_patch_embeddings, attention_mask
    )

    # Forward pass through language model
    language_model_output = self.language_model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        labels=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )

    # Extract hidden states for action tokens
    last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
    actions_hidden_states = last_hidden_states[
        :,
        NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
        :,
    ]  # (B, act_chunk_len, D)

    # Handle different prediction methods
    if action_head is not None:
        # L1 regression prediction
        normalized_actions = action_head.predict_action(actions_hidden_states)
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        normalized_actions = normalized_actions.float().cpu().detach().numpy()
    else:
        # Discrete token-based prediction
        predicted_action_token_ids = (
            language_model_output.logits[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            ]
            .argmax(dim=2)
            .cpu()
            .numpy()
        )
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

    return normalized_actions, actions_hidden_states
```

# 实验

|**任务**|**LIBERO-10**|**LIBERO-goal**|**LIBERO-object**|**LIBERO-spatial**|
|---|---|---|---|---|
|**OpenVLA-OFT复现**|95.2%|97.2%|99.4%|98.4%|
|**OpenVLA-OFT原文**|94.5%|97.9%|98.4%|97.6%|


