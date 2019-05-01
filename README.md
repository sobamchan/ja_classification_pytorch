## Prepare dataset
```bash
python baseline/data.py --dpath ./datas --savedir ./datas/datasets
```

## Train
```bash
python baseline/train.py run --dataset-dir ./datas/datasets
```

## Evaluate
```bash
curl -d "@pred.txt" -X POST 172.21.65.41:5000
```
