import numpy as np, pandas as pd, time, tracemalloc, hashlib, os
import dask.dataframe as dd
rng=np.random.default_rng(42); N=3_000_000
if not os.path.exists("applicants.csv"):
    g=rng.choice(["A","B"],N,p=[0.6,0.4])
    score=rng.normal(70,10,N)-np.where(g=="B",4,0)  # historical bias baked into the score
    df=pd.DataFrame({"applicant_id":np.arange(N),"email":[f"user{i}@mail.com" for i in range(N)],
        "group":g,"years_exp":rng.integers(0,30,N),"score":score.round(2)})
    df["shortlisted"]=(df.score>75).astype(int)
    df.to_csv("applicants.csv",index=False)
print("size MB", os.path.getsize("applicants.csv")/1e6)
# full load
tracemalloc.start(); t=time.time()
df=pd.read_csv("applicants.csv"); r_full=df.groupby("group").shortlisted.mean()
full=(time.time()-t, tracemalloc.get_traced_memory()[1]/1e6); tracemalloc.stop(); del df
# chunked, with pseudonymisation + minimization
salt=b"s3cret"
tracemalloc.start(); t=time.time(); agg=None
for ch in pd.read_csv("applicants.csv",chunksize=250_000, usecols=["applicant_id","group","shortlisted"]):
    ch["pid"]=ch.applicant_id.astype(str).map(lambda x: hashlib.sha256(salt+x.encode()).hexdigest()[:16])
    ch=ch.drop(columns="applicant_id")
    s=ch.groupby("group").shortlisted.agg(["sum","count"])
    agg=s if agg is None else agg.add(s,fill_value=0)
r_chunk=agg["sum"]/agg["count"]
chunk=(time.time()-t, tracemalloc.get_traced_memory()[1]/1e6); tracemalloc.stop()
# dask
t=time.time()
ddf=dd.read_csv("applicants.csv",blocksize="32MB",usecols=["group","shortlisted"])
r_dask=ddf.groupby("group").shortlisted.mean().compute()
dt=time.time()-t
print("full",full,r_full.to_dict()); print("chunk",chunk,r_chunk.to_dict()); print("dask",dt,r_dask.to_dict(), ddf.npartitions)
print("DI ratio", r_dask["B"]/r_dask["A"])
