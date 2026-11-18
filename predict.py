"""Score new applications with a model bundle created by main.py."""
import argparse
from pathlib import Path
import joblib, numpy as np, pandas as pd
def positive_probability(bundle, frame):
    """Return the probability for the model class trained as the chosen positive target."""
    model = bundle["pipeline"]
    positive_class = bundle.get("positive_model_class", 1)
    indices = np.flatnonzero(np.asarray(model.classes_) == positive_class)
    if len(indices) != 1:
        raise ValueError(f"Model does not expose expected positive class {positive_class!r}")
    return model.predict_proba(frame)[:, int(indices[0])]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model",default="outputs/model.joblib");p.add_argument("--input",required=True);p.add_argument("--out",default="outputs/scored_applications.csv");a=p.parse_args()
 bundle=joblib.load(a.model);df=pd.read_csv(a.input);features=bundle["features"]
 missing=[c for c in features if c not in df]
 if missing:raise SystemExit(f"Scoring file is missing training features: {missing}")
 labels=np.asarray(bundle["classes"]);pos=bundle["positive_label"]
 # main.py remaps the selected positive label to binary class 1 before fitting.
 prob=positive_probability(bundle,df[features]);threshold=float(bundle["threshold"])
 keep=[c for c in ["application_id","customer_id","id"] if c in df]
 out=pd.DataFrame({c:df[c] for c in keep});out["default_probability"]=prob;out["review_flag"]=prob>=threshold
 path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);out.to_csv(path,index=False)
 print(f"Scored {len(out):,} rows; flagged {int(out.review_flag.sum()):,} for human review. Output: {path}")
if __name__=="__main__":main()
