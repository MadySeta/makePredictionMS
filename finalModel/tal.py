from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import pandas

df = pandas.read_csv("/home-local/ahuyghe/Documents/SDBD/tal/twitter_human_bots_dataset.csv")
df["description"] = df["description"].fillna("")

df_short = df

# def equilibrate_dataset(df, column_name, minority_proportion=1):
#     # Trouver la classe majoritaire et minoritaire
#     counts = df[column_name].value_counts()
#     majority_class = counts.idxmax()
#     minority_class = counts.idxmin()
    
#     # Séparer les données en classes majoritaire et minoritaire
#     majority_data = df[df[column_name] == majority_class]
#     minority_data = df[df[column_name] == minority_class]
    
#     # Déterminer le nombre d'échantillons de la classe majoritaire à conserver
#     minority_count = len(minority_data)
#     majority_count = int(minority_count * minority_proportion)
    
#     # Échantillonner aléatoirement la classe majoritaire avec un nombre restreint
#     majority_sampled = majority_data.sample(n=majority_count, random_state=42)
    
#     # Concaténer les données équilibrées
#     balanced_df = pandas.concat([majority_sampled, minority_data])
    
#     return balanced_df

# balanced_df = equilibrate_dataset(df_short, 'account_type', minority_proportion=1)

balanced_df = df_short

# Afficher les informations sur les nouvelles classes équilibrées
print(balanced_df['account_type'].value_counts())

v = TfidfVectorizer()
transformed_df = v.fit_transform(balanced_df["description"])

# feature_names = v.get_feature_names_out()
# for i, word in enumerate(feature_names) : 
#     indx = v.vocabulary_.get(word)
#     print(f"{i} {word} {v.idf_[indx]}")
# print(df_short[26])
# print(transformed_df[26])
# print(v.vocabulary_.get("that"))

# X_train, X_test, y_train, y_test = train_test_split(
#     transformed_df
# )

for i in range(transformed_df.shape[0]):
    transformed_df[i].toarray()[0] 


X_train, X_test, y_train, y_test = train_test_split(
    balanced_df.description,
    balanced_df.account_type,
    test_size = 0.2,
    random_state = 42,
    stratify=balanced_df.account_type
)

print(y_train.value_counts())
print(y_test.value_counts())

clf = Pipeline([
     ('vectorizer_tfidf',TfidfVectorizer()),    
     ('KNN', KNeighborsClassifier())         
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

for i in range(len(y_test[:200])):
    print(f"{X_test.iloc[i]} \n\n test :{y_test.iloc[i]} || pred :{y_pred[i]}\n\n")
