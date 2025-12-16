# chess-teacher

[Notion](https://chain-culotte-70e.notion.site/1-Real-world-Chess-Robot-Teacher-27ffb979596d81f4a366d3dfc53976bc?source=copy_link) [Video](https://youtu.be/p1ADruJ302M) 

## Dataset Generation

1. Activate Conda environment:

```bash
conda activate gr00t
```

2. Launch teleoperation nodes

```bash
cd RFM/teleop
python3 quest_publisher.py
```

```bash
cd RFM/teleop
python3 fr3_controller.py
```

3. Start recording the dataset

```bash
cd RFM/teleop
bash record.sh
```

Recorded episodes are saved in `RFM/teleop/episodes`.

## VLM Reasoning

1. Install required library

```bash
pip install huggingface
```

2. download VLM inference model (will take about 5 mins)

```bash
hf download Qwen/Qwen3-VL-4B-Thinking
```

3. Run sample reasoning code

```bash
python3 policy/vlm_reasoning.py
```
