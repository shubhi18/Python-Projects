def replace(df):
    Q = df.isna().sum()
    missing_columns = list(Q[Q>0].index)
    for i in missing_columns:
        if(df[i].dtypes == "object"):
            x = df[i].mode()[0]
            df[i] = df[i].fillna(x)
        else:
            x = df[i].mean()
            df[i] = df[i].fillna(x)
            

def catcon(df):
    cat=[]
    con=[]
    for i in df.columns:
        if df[i].dtypes=="object":
            cat.append(i)
        else:
            con.append(i)
    return cat,con


def replacer(df):
    cat,con=catcon(df)
    for i in cat:
        t=df[i].mode()[0]
        df[i]=df[i].fillna(t)

    for i in con:
        t=df[i].mean()
        df[i]=df[i].fillna(t)
        
def standardize(df):
    cat,con=catcon(df)
    from sklearn.preprocessing import StandardScaler
    ss=StandardScaler()
    import pandas as pd
    Q=pd.DataFrame(ss.fit_transform(df[con]) , columns = con)
    return Q

def preprocessing(df):
    cat,con = catcon(df)
    from sklearn.preprocessing import MinMaxScaler
    ss = MinMaxScaler()
    import pandas as pd
    X1 = pd.DataFrame(ss.fit_transform(df[con]),columns=con)
    X2 = pd.get_dummies(df[cat])
    Xnew = X1.join(X2)
    return Xnew
    
def outlier(df):
    out = []
    cat,con = catcon(df)
    df = standardize(df)
    for i in con:
        out.extend(list(df[(df[i]>3)|(df[i]<-3)].index))

    import numpy as np
    outliers = list(np.unique(out))
    return outliers


def ANOVA(df,cat,con):
    from statsmodels.formula.api import ols
    eqn = str(con) + " ~ " + str(cat)
    model = ols(eqn,df).fit()
    from statsmodels.stats.anova import anova_lm
    Q = anova_lm(model)
    return round(Q.iloc[0:1,4:5].values[0][0],5)


def chisq(df,cat1,cat2):
    import pandas as pd
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(df[cat1],df[cat2])
    a,b,c,d = chi2_contingency(ct)
    return round(b,5)


