"""Stage 0: read the raw TSVs (tab-separated, no quoting) and cache them as parquet."""
import sys
data, work = sys.argv[1], sys.argv[2]
import pandas as pd, csv
for split in ["train","test"]:
    for s in [1,2,3]:
        p=f"{data}/{split}/{split}_source{s}.tsv"
        df=pd.read_csv(p,sep="\t",dtype=str,keep_default_na=False,quoting=csv.QUOTE_NONE)
        df.to_parquet(f"{work}/{split}_s{s}.parquet",index=False)
        print(p,df.shape)
gt=pd.read_csv(f"{data}/train/train_ground_truth.tsv",sep="\t",dtype=str,keep_default_na=False,quoting=csv.QUOTE_NONE)
gt.to_parquet(f"{work}/train_gt.parquet",index=False); print(gt.shape)
