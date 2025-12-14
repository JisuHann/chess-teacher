import os
from PIL import Image

LLM_CONFIG = {
    'dtype': 'bfloat16',
    'device_map': 'cuda',
}

def load_vlm(model_name: str):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    tokenizer = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **LLM_CONFIG)
    return tokenizer, model

def format_messages(args, question: str, image_path: str):
    # system prompt
    system_path ="src/vlm/prompt/system.txt"
    with open(system_path, 'r') as f:
        system_message = f.read()

    # user (fewshot, prompt)
    image = Image.open(image_path).convert("RGB")
    user_message = [{"type": "text","text": question}]
    if image_path:
        user_message.append({"type": "image", "image":image})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content":user_message},
    ]
    return messages

def inference(model, processor, messages: list):
    # Pre-process
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, 
                                    max_new_tokens=8192,
                                    do_sample=False,
                                    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(output_text)
    reasoning, final_answer = output_text[0].split("Final Answer: ")
    return reasoning, final_answer