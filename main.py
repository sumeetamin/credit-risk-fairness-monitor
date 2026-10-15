"""Credit default risk baseline, decision-threshold analysis, and subgroup audit."""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             balanced_accuracy_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True); p.add_argument("--target",required=True)
    p.add_argument("--group",default=None,help="Optional sensitive or operational group for error-rate audit")
    p.add_argument("--positive-label",default=None); p.add_argument("--threshold",type=float,default=0.5)
    p.add_argument("--out",default="outputs"); a=p.parse_args()
    df=pd.read_csv(a.input)
    if a.target not in df: raise SystemExit(f"Missing target column: {a.target}")
    group=df[a.group].astype("string") if a.group and a.group in df else None
    yraw=df[a.target]; X=df.drop(columns=[a.target])
    if a.group and a.group in X: X=X.drop(columns=[a.group])
    # Remove common row identifiers; they are not meaningful predictors.
    X=X.drop(columns=[c for c in X if c.lower() in {"id","customerid","member_id","encounter_id"}],errors="ignore")
    labels=LabelEncoder().fit(yraw.astype(str)); y=labels.transform(yraw.astype(str))
    if len(labels.classes_)!=2: raise SystemExit("This baseline expects exactly two target classes.")
    pos=str(a.positive_label) if a.positive_label is not None else str(labels.classes_[-1])
    if pos not in labels.classes_: raise SystemExit(f"positive-label must be one of {list(labels.classes_)}")
    pos_id=int(np.where(labels.classes_==pos)[0][0])
    numeric=X.select_dtypes(include="number").columns.tolist()
    categorical=[c for c in X.columns if c not in numeric]
    prep=ColumnTransformer([
        ("numeric",Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler())]),numeric),
        ("categorical",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),categorical)
    ])
    model=Pipeline([("features",prep),("classifier",LogisticRegression(max_iter=1500,class_weight="balanced"))])
    Xtr,Xte,ytr,yte,idxtr,idxte=train_test_split(X,y,np.arange(len(y)),test_size=.25,random_state=42,stratify=y)
    model.fit(Xtr,ytr); prob=model.predict_proba(Xte)[:,pos_id]; pred=(prob>=a.threshold).astype(int); truth=(yte==pos_id).astype(int)
    tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
    metrics={"rows":int(len(df)),"train_rows":int(len(Xtr)),"test_rows":int(len(Xte)),
             "positive_label":pos,"threshold":a.threshold,
             "roc_auc":float(roc_auc_score(truth,prob)),"average_precision":float(average_precision_score(truth,prob)),
             "brier_score":float(brier_score_loss(truth,prob)),"balanced_accuracy":float(balanced_accuracy_score(truth,pred)),
             "confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}}
    audit=[]
    if group is not None:
        g=group.iloc[idxte].fillna("UNKNOWN").astype(str).to_numpy()
        for val in sorted(set(g)):
            m=g==val; actual=truth[m]; guess=pred[m]
            if m.sum()<10: continue
            tpr=float(((guess==1)&(actual==1)).sum()/max(1,(actual==1).sum()))
            fpr=float(((guess==1)&(actual==0)).sum()/max(1,(actual==0).sum()))
            audit.append({"group":val,"n":int(m.sum()),"positive_rate":float(guess.mean()),"tpr":tpr,"fpr":fpr})
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    metrics["group_audit"]=audit
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    pd.DataFrame({"actual":labels.inverse_transform(yte),"probability_positive":prob,"predicted":np.where(pred==1,pos,labels.classes_[1-pos_id])}).to_csv(out/"holdout_predictions.csv",index=False)
    joblib.dump({"pipeline":model,"classes":labels.classes_.tolist(),"positive_label":pos,"threshold":a.threshold},out/"model.joblib")
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
