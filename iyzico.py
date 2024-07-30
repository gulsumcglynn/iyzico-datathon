import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import  mean_absolute_error
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

#train ve submission datasetlerini düzenleyip birleştiriyoruz.
df_train = pd.read_csv("datasets/train.csv", parse_dates=["month_id"])
df_sub = pd.read_csv( "datasets/sample_submission.csv.zip")
df_sub['month_id'] = df_sub['id'].str.extract(r'(\d{6})')
id_splitted = df_sub['id'].str.split('merchant_',n=1)
df_sub['id'] ='merchant_'+id_splitted.str[1]
df_sub =df_sub.drop("id", axis=1)
df_sub =df_sub[["month_id","net_payment_count"]]

df = pd.merge(df_train, df_sub, on=["month_id","net_payment_count"], how="outer")
df["month_id"] = pd.to_datetime(df["month_id"], format='%Y%m')

#merchant_id Maskelenmiş iş yeri ID'si
#month_id İşlemin yapıldığı ay (YYYYMM formatında)
#merchant_source İş yerinin iyzico’ya katıldığı kaynak(3 kaynak var)
#settlement_period İş yerinin hak edişini alış sıklığı(3 periyot var)
#working_type İş yerinin tipini gösterir(6 iş yeri tipi var)
#mcc_id İş yerinin satış yaptığı kategori bilgisini gösterir (172 kategori bilgisi var)
#merchant_segment İş yerinin iyzico içerisinde bulunduğu segmenti gösterir(4 segment var)
#net_payment_count İş yerinin ilgili ay içerisinde geçirdiği net (ödeme - iptal - iade) işlem sayısıdır.
#train 2020-01 den 2023-09 submission 2023-10 den 2023-12

#DATASETİN GENEL RESMİ
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)
df.head()
df.info()

#KATEGORİK NUMERIK DEĞİŞKENLERİ AYRIŞTIRMA
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols
cat_cols, cat_but_car, num_cols = grab_col_names(df)

##################################
# AYKIRI DEĞER ANALİZİ
##################################
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
outlier_thresholds(df, num_cols)

#net_payment_count (-) ve (0) da olması olağan olduğu için baskılamadık.
# alt limit -193.5 üst limit 322.5

#VERİ GÖRSELLEŞTİRME

######################################
#Hedef Değişken Analizi (Analysis of Target Variable)
######################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df,"net_payment_count",col)


# Bağımlı değişkenin ay bazında incelenmesi
df.set_index('month_id', inplace=True)
a = df["net_payment_count"].resample("MS").mean()
a.plot(figsize=(15, 6))
plt.show()

# Nadir sınıfların tespit edilmesi ve rar encoder yapılması
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "net_payment_count", cat_cols)
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df
rare_encoder(df,0.01)

#FEATURE ENGINEERING

df["busy_working_type_segment"] = ((df["working_type"] == "Working Type - 5") | (df["working_type"] == "Working Type - 6")) & (df["merchant_segment"] == "Segment - 4")
df["busy_working_type_source"] = ((df["working_type"] == "Working Type - 5") | (df["working_type"] == "Working Type - 6")) & (df["merchant_source_name"] == "Merchant Source - 1")
df["medium_busy_working_type_source"] = ((df["working_type"] == "Working Type - 5") | (df["working_type"] == "Working Type - 6")) & (df["merchant_source_name"] == "Merchant Source - 2")
df["busy_working_type_period"] = ((df["working_type"] == "Working Type - 5") | (df["working_type"] == "Working Type - 6")) & (df["settlement_period"] == "Settlement Period - 1")

df["busy_working_type"] = ((df["working_type"] == "Working Type - 5") | (df["working_type"] == "Working Type - 6"))
df["not_busy_working_type"] = (~(df["working_type"] == "Working Type - 5") | (df["working_type"] == "Working Type - 6"))

df["busy_period_segment"] = ((df["settlement_period"] == "Settlement Period - 1") & (df["merchant_segment"] == "Segment - 4"))
df["busy_source_segment"] = ((df["merchant_source_name"] == "Merchant Source - 1") & (df["merchant_segment"] == "Segment - 4"))
df["medium_busy_source_segment"] = ((df["merchant_source_name"] == "Merchant Source - 2") & (df["merchant_segment"] == "Segment - 4"))

df["busy_source_period"] = ((df["merchant_source_name"] == "Merchant Source - 1") & (df["settlement_period"] == "Settlement Period - 1"))
df["medium_busy_source_period"] = ((df["merchant_source_name"] == "Merchant Source - 2") & (df["settlement_period"] == "Settlement Period - 1"))
df["low_gain_times"] = (df["month_id"] == "2020-12-01") & (df["month_id"] == "2021-12-01") & (df["month_id"] == "2022-12-01")

#Zaman değişkenleri ekledim.
def create_date_features(df, date_column):
    df['month'] = df[date_column].dt.month
    df['day_of_month'] = df[date_column].dt.day
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['year'] = df[date_column].dt.year
    df["is_wknd"] = df[date_column].dt.weekday // 4
    df['is_month_start'] =df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['quarter'] = df[date_column].dt.quarter
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    return df

df = create_date_features(df, "month_id")

#Dataframe e gecikme ve hareketli ortalamalar ile yeni değişkenler ekledim.
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["merchant_id"])['net_payment_count'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 182, 272, 362])
#tahminimiz 3 aylık olduğu için 3 ay ve katlarına göre gecikme ekliyoruz.
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['payment_roll_mean_' + str(window)] = dataframe.groupby(["merchant_id"])['net_payment_count']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [91, 120, 152, 182, 242, 402, 542, 722])

##################
# Label Encoding & One-Hot Encoding İşlemler.
##################

cat_cols, cat_but_car, num_cols = grab_col_names(df)

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

for col in cat_cols:
    label_encoder(df, col)

df = one_hot_encoder(df, cat_cols, drop_first=True)


cat_cols, num_cols, cat_but_car = grab_col_names(df)


##################################
# MODELLEME
##################################

train = df[df["net_payment_count"].notnull()]
test = df[df["net_payment_count"].isnull()]

y = train['net_payment_count']
X = train.drop(["net_payment_count", "merchant_id", "mcc_id", "month_id"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

xgboost = XGBRegressor().fit(X_train, y_train)
y_pred = xgboost.predict(X_test)
mean_absolute_error(y_test, y_pred)

################################################################
# Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını görselleştirdim.
################################################################

# feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = XGBRegressor()
model.fit(X, y)

plot_importance(xgboost, X, num=20)


########################################
#Submission dataframeinde ki net_payment_count değerlerini tahmin ettim ve final_df'i csv formatına çevirdim.
########################################

nan_rows = df[df["month_id"] >= "2023-10-01"]
submission_df = nan_rows[['month_id', 'net_payment_count']]
model = XGBRegressor()
model.fit(X, y)
new_df = df[df["month_id"] >= "2023-10-01"].drop(["mcc_id", "merchant_id", "net_payment_count", "month_id"], axis=1)
predictions = model.predict(new_df)
dictionary = {"month_id": submission_df["month_id"], "net_payment_count2": predictions}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.reset_index(inplace=True)
dfSubmission.drop(["index", "month_id"], axis=1, inplace=True)

df_sub = pd.read_csv( "datasets/sample_submission.csv.zip")
final_df = dfSubmission.join(df_sub, how='left')
final_df.drop(["net_payment_count"], inplace=True, axis=1)

final_df.rename(columns={'net_payment_count2': 'net_payment_count'}, inplace=True)
final_df.to_csv("submission_df.csv", index=False)



