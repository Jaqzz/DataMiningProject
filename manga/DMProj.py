import pandas as pd
import ast as ast
import collections
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv('C:/Users/Elijah/Desktop/manga/manga.csv')
#not for a school project, thanks
df = df[df['sfw'] != False]
#df2 = df

for i in df.index:
    df['genres'][i] = df['genres'][i].strip('][').split(', ')
lol = df['genres'].values.tolist()

#This code taken heavily from mlxtend documentation at https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
#formatting input
te = TransactionEncoder()
te_ary = te.fit(lol).transform(lol)
tfdf = pd.DataFrame(te_ary, columns=te.columns_)

apdf = apriori(tfdf, min_support=0.03, use_colnames=True)
ardf = association_rules(apdf, min_threshold=0.1)
ardf.to_csv('C:/Users/Elijah/Desktop/manga/association.csv')

score = 0
aud = 0
count = 0
topmanga = []
avgscore = []
genrecount = []
avgaud = []
found = False

df['score'] = df['score'].fillna(0)
for x in apdf.index:
    apdf['itemsets'][x] = list(apdf['itemsets'][x])
    for y in df.index:
        #to eliminate entries with no scores skewing the data
        if int(df['score'][y]) > 5:
            if all([item in df['genres'][y] for item in apdf['itemsets'][x]]):
                score = score + int(df['score'][y])
                aud = aud + int(df['members'][y])
                count += 1
                #Find the highest rated manga for that genre. Only works because the original csv is sorted by score.
                if found == False:
                    title = df['title'][y]
                    found = True
    avgs = score / count
    topmanga.append(title)
    avgscore.append(avgs)
    avga = aud / count
    avgaud.append(avga)
    genrecount.append(count)
    found = False
    score = 0
    aud = 0
    count = 0
apdf['highest_rated_title'] = topmanga
apdf['average_score'] = avgscore
apdf['average_audience'] = avgaud
apdf['genre_count'] = genrecount
apdf.to_csv('C:/Users/Elijah/Desktop/manga/apriori.csv')