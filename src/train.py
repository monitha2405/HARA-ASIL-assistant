"""EDA -> preprocessing -> feature eng/selection/SVD -> model comparison -> tuning -> evaluation -> save."""
import json, time, pickle, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from .data_gen import generate, NUM
from .asil import asil, ORDER
SEED=42; np.random.seed(SEED)

def make_pipe(clf):
    pre = ColumnTransformer([
        ("txt", Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=2)),("svd",TruncatedSVD(40,random_state=SEED))]),"text"),
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),NUM)])
    return Pipeline([("pre",pre),("sel",SelectKBest(f_classif,k=30)),("clf",clf)])
MODELS={"LogReg":(LogisticRegression(max_iter=3000),{"clf__C":[.1,1,10]}),
 "SVM":(SVC(probability=True,random_state=SEED),{"clf__C":[1,10]}),
 "RandomForest":(RandomForestClassifier(200,random_state=SEED,n_jobs=-1),{"clf__max_depth":[None,12]}),
 "kNN":(KNeighborsClassifier(),{"clf__n_neighbors":[5,11,21]})}

def eda(df):
    sns.set_theme(style="dark")
    fig,ax=plt.subplots(1,4,figsize=(16,3.5))
    for a,c in zip(ax,["S","E","C","ASIL"]):
        (df[c].value_counts().reindex(ORDER if c=="ASIL" else sorted(df[c].unique()))).plot.bar(ax=a,color="#ffb000"); a.set_title(c+" distribution")
    plt.tight_layout(); plt.savefig("reports/eda_class_distribution.png",dpi=110); plt.close()
    plt.figure(figsize=(6,5)); sns.heatmap(df[NUM+["S","E","C"]].corr(),annot=True,fmt=".2f",cmap="vlag"); plt.title("Correlation"); plt.tight_layout()
    plt.savefig("reports/eda_correlation.png",dpi=110); plt.close()
    X=SimpleImputer(strategy="median").fit_transform(df[NUM]); X=StandardScaler().fit_transform(X)
    p=PCA(2).fit(X); Z=p.transform(X)
    plt.figure(figsize=(6,5)); sns.scatterplot(x=Z[:,0],y=Z[:,1],hue=df.ASIL,hue_order=ORDER,s=12,palette="viridis")
    plt.title(f"PCA of numeric features (var {p.explained_variance_ratio_.sum():.0%})"); plt.tight_layout(); plt.savefig("reports/eda_pca.png",dpi=110); plt.close()

def main():
    t0=time.time(); df=generate(); df.to_csv("data/hara_synthetic.csv",index=False)
    n0=len(df); df=df.drop_duplicates(subset=["text","speed_kmh","exposure_pct","reaction_s"]).reset_index(drop=True)
    eda(df); feats=["text"]+NUM
    tr,te=train_test_split(df,test_size=.2,stratify=df.ASIL,random_state=SEED)
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED); out={"models":{},"best":{}}; fitted={}; pred=pd.DataFrame(index=te.index)
    for t in ["S","E","C"]:
        base=cross_val_score(make_pipe(DummyClassifier(strategy="most_frequent")),tr[feats],tr[t],cv=cv,scoring="f1_macro").mean()
        scores={k:float(cross_val_score(make_pipe(m),tr[feats],tr[t],cv=cv,scoring="f1_macro").mean()) for k,(m,_) in MODELS.items()}
        scores["Baseline"]=float(base); out["models"][t]=scores
        best=max(MODELS,key=scores.get); m,g=MODELS[best]
        gs=GridSearchCV(make_pipe(m),{**g,"sel__k":[20,30,"all"]},cv=cv,scoring="f1_macro",n_jobs=-1).fit(tr[feats],tr[t])
        fitted[t]=gs.best_estimator_; pred[t]=gs.predict(te[feats])
        out["best"][t]={"model":best,"params":{k:str(v) for k,v in gs.best_params_.items()},"cv_f1":float(gs.best_score_),
            "test_acc":float(accuracy_score(te[t],pred[t])),"test_f1":float(f1_score(te[t],pred[t],average="macro"))}
        print(t,best,out["best"][t])
    pa=[asil(*x) for x in zip(pred.S,pred.E,pred.C)]; ta=list(te.ASIL); ri={a:i for i,a in enumerate(ORDER)}
    under=np.mean([ri[p]<ri[a] for p,a in zip(pa,ta)]); over=np.mean([ri[p]>ri[a] for p,a in zip(pa,ta)])
    out["asil"]={"accuracy":float(accuracy_score(ta,pa)),"macro_f1":float(f1_score(ta,pa,average="macro")),
        "under_classification_rate":float(under),"over_classification_rate":float(over),
        "within_one_level":float(np.mean([abs(ri[p]-ri[a])<=1 for p,a in zip(pa,ta)]))}
    cm=confusion_matrix(ta,pa,labels=ORDER); plt.figure(figsize=(5.5,4.5)); sns.heatmap(cm,annot=True,fmt="d",xticklabels=ORDER,yticklabels=ORDER,cmap="YlOrBr")
    plt.xlabel("Predicted ASIL"); plt.ylabel("True ASIL"); plt.tight_layout(); plt.savefig("reports/confusion_asil.png",dpi=110); plt.close()
    out["meta"]={"n_rows":int(n0),"n_after_dedup":int(len(df)),"train":len(tr),"test":len(te),"train_seconds":round(time.time()-t0,1),
        "data":"synthetic (see data_gen.py)","version":time.strftime("%Y%m%d-%H%M")}
    print(classification_report(ta,pa,labels=ORDER,zero_division=0))
    pickle.dump(fitted,open("models/hara_model.pkl","wb")); json.dump(out,open("models/metrics.json","w"),indent=2); print(out["asil"])
    from . import rag; n,emb=rag.build_library(); print(f"RAG library: {n} approved hazards, embedder={emb}")
if __name__=="__main__": main()
