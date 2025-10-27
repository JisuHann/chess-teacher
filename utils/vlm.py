

QWEN_MODEL = ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]
LLM_CONFIG = {
    'torch_dtype': 'bfloat16',
    'device_map': 'cuda',
}
def load_vlm(model_name: str):
    
    if model_name not in QWEN_MODEL:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **LLM_CONFIG)
    if model_name in QWEN_MODEL:
        from transformers import Qwen2_5_VLForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **LLM_CONFIG)
    return tokenizer, model

def format_messages(args, question: str, image_path: str):
    # system prompt
    with open(system_path, 'r') as f:
        system_message = f.read()

    # user (fewshot, prompt)
    if args.few_shot:
        with open(os.path.join(args.prompt_path, 'few_shot.txt'), 'r') as f:
            few_shot_examples    = f.read()
        user_message = [{"type": "text","text": few_shot_examples}]
    else:
        user_message = []
    user_message = [{"type": "text","text": question}]
    if image_path:
        user_message.append({"type": "image", "image":f"file://{image_path}"s})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content":user_message},
    ]
    return messages

def inference(model, processor, messages: list):
    # Pre-process
    from qwen_vl_utils import process_vision_info
    input_ids = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor([input_ids], images = image_inputs, padding = True, return_tensors = "pt").to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response