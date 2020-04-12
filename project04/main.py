# 引入必要的包
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# 指定数据集路径
dataset_path = './data'

# 加载数据
book1_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book1-edges.csv'))
book2_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book2-edges.csv'))
book3_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book3-edges.csv'))
book4_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book4-edges.csv'))
book5_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book5-edges.csv'))

book1_df.head()

# 从dataframe构建网络
G_book1 = nx.from_pandas_dataframe(book1_df, 'Source', 'Target', edge_attr=['weight', 'book'])
G_book2 = nx.from_pandas_dataframe(book2_df, 'Source', 'Target', edge_attr=['weight', 'book'])
G_book3 = nx.from_pandas_dataframe(book3_df, 'Source', 'Target', edge_attr=['weight', 'book'])
G_book4 = nx.from_pandas_dataframe(book4_df, 'Source', 'Target', edge_attr=['weight', 'book'])
G_book5 = nx.from_pandas_dataframe(book5_df, 'Source', 'Target', edge_attr=['weight', 'book'])

G_books = [G_book1, G_book2, G_book3, G_book4, G_book5]


G_book1.edges(data=True)


# 简单可视化
plt.figure(figsize=(10, 9))
nx.draw_networkx(G_book1)
plt.show()


# degree centrality

# 计算每个网络的 degree centrality
deg_cent_list = [nx.degree_centrality(G_book) for G_book in G_books]

# 将计算结果构建成Series
deg_cent_series_list = [pd.Series(deg_cent) for deg_cent in deg_cent_list]


top_n = 10

for i, deg_cent_series in enumerate(deg_cent_series_list):
    print('第{}本书最重要的{}个人物：'.format(i + 1, top_n))
    # 取出每个图的 top_n 个节点
    top_characters = deg_cent_series.sort_values(ascending=False)[:top_n]
    print(top_characters)

    print()

# betweenness centrality

# 计算每个网络的 betweenness centrality
btw_cent_list = [nx.betweenness_centrality(G_book) for G_book in G_books]

# 将计算结果构建成Series
btw_cent_series_list = [pd.Series(btw_cent) for btw_cent in btw_cent_list]



top_n = 10

for i, btw_cent_series in enumerate(btw_cent_series_list):
    print('第{}本书最重要的{}个人物：'.format(i + 1, top_n))
    # 取出每个图的 top_n 个节点
    top_characters = btw_cent_series.sort_values(ascending=False)[:top_n]
    print(top_characters)

    print()
    


# page rank

# 计算每个网络的 page rank
page_rank_list = [nx.pagerank(G_book) for G_book in G_books]

# 将计算结果构建成Series
page_rank_series_list = [pd.Series(page_rank) for page_rank in page_rank_list]

top_n = 10

for i, page_rank_series in enumerate(page_rank_series_list):
    print('第{}本书最重要的{}个人物：'.format(i + 1, top_n))
    # 取出每个图的 top_n 个节点
    top_characters = page_rank_series.sort_values(ascending=False)[:top_n]
    print(top_characters)

    print()
    


cor_df = pd.DataFrame(columns=['Degree Centrality', 'Closeness Centrality', 'Betweenness Centrality', 'Page Rank'])
cor_df['Degree Centrality'] = pd.Series(nx.degree_centrality(G_book1))
cor_df['Closeness Centrality'] = pd.Series(nx.closeness_centrality(G_book1))
cor_df['Betweenness Centrality'] = pd.Series(nx.betweenness_centrality(G_book1))
cor_df['Page Rank'] = pd.Series(nx.pagerank(G_book1))
cor_df.head()



trend_df = pd.DataFrame(columns=['Book1', 'Book2', 'Book3', 'Book4', 'Book5'])
trend_df['Book1'] = pd.Series(nx.degree_centrality(G_book1))
trend_df['Book2'] = pd.Series(nx.degree_centrality(G_book2))
trend_df['Book3'] = pd.Series(nx.degree_centrality(G_book3))
trend_df['Book4'] = pd.Series(nx.degree_centrality(G_book4))
trend_df['Book5'] = pd.Series(nx.degree_centrality(G_book5))
trend_df.fillna(0, inplace=True)
trend_df.head()


# 第1本书中最重要top10人物的趋势
top_10_from_book1 = trend_df.sort_values('Book1', ascending=False)[:10]
top_10_from_book1

top_10_from_book1.T.plot(figsize=(10, 8))
plt.tight_layout()


plt.figure(figsize=(15, 10))

# 节点颜色由节点的度决定
node_color = [G_book5.degree(v) for v in G_book5]

# 节点的大小由degree centrality决定
node_size = [10000 * nx.degree_centrality(G_book5)[v] for v in G_book5]

# 边的宽度由权重决定
edge_width = [0.2 * G_book5[u][v]['weight'] for u, v in G_book5.edges()]

# 使用spring布局
pos=nx.spring_layout(G_book5)

nx.draw_networkx(G_book5, pos, node_size=node_size, 
                 node_color=node_color, alpha=0.7, 
                 with_labels=False, width=edge_width)

# 取出第一本书的top10人物
top10_in_book1 = top_10_from_book1.index.values.tolist()
# 构建label
labels = {role: role for role in top10_in_book1}

# 给网络添加label
nx.draw_networkx_labels(G_book5, pos, labels=labels, font_size=10)

plt.axis('off')
plt.tight_layout()
plt.show()
