# chess-teacher

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
Recorded episodes are saved in ```RFM/teleop/episodes```.