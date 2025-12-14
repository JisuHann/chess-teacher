import argparse
from vlm import utils as vlm

def inference_chess_teacher(args):
    tokenizer, model = vlm.load_vlm(args.model_name)
    messages = vlm.format_messages(args, args.question, args.image_path)
    reasoning, response = vlm.inference(model, tokenizer, messages)
    print("Reasoning: ", reasoning)
    print("Task chosen: ", response)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", default="utils/prompt", type=str) 
    # parser.add_argument("--question", default="I am the coach. Come up with a best chess move for the given chess board.", type=str)
    parser.add_argument("--question", default="I am the player. Come up with a best chess move for the given chess board.", type=str)
    parser.add_argument("--image_path", default="src/vlm/asset/chess.jpeg", type=str)
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Thinking", type=str)
    args = parser.parse_args()
    inference_chess_teacher(args)
