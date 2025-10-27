import huggingface
from utils import vlm

def inference_chess_teacher(question: str, model_name: str):
    tokenizer, model = vlm.load_vlm(model_name)
    messages = vlm.format_messages(args, args.question, args.image_path)
    response = vlm.inference(messages)
    return

if __name__ == "__main__":
    args = parse_args()
    parser.add_argument("--prompt_path", default="utils/prompt", type=str, required=True)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--question", default="", type=str, required=True)
    parser.add_argument("--image_path", default="", type=str, required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct", type=str, required=True)
    main(args)