# -*- coding: utf-8 -*-
"""SUBMISSION 2 ML TERAPAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jIuMQ0sjilLoyclXDFWrbGiwHnZgziWn

# **IMPORT LIBRARY**

Pada bagian ini proses penginstallan library demoji dilakukan.
"""
"""Semua library yang dibutuhkan selama pelatihan model diimport"""

import pandas as pd
import numpy as np
from zipfile import ZipFile
import demoji
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

"""# **GATHERING/LOAD DATA**

Pada proyek kali ini, digunakan 2 dataset. Dataset pertama diberi nama 'rating', dan dataset kedua diberi nama 'blog'.
"""

rating = pd.read_csv('Blog Ratings.csv')
blog = pd.read_csv('Medium Blog Data.csv')

"""# **EXPLORATORY DATA ANALYSIS**

### **Univariate Analysis**
"""

rating.info()

"""Pada code diatas ditemukan bahwa pada dataset 'rating' memiliki 3 kolom, yaitu 'blog_id', 'userId', dan 'ratings'. Masing-masing kolom tersebut memiliki data sebanyak 200,140 data"""

print('Jumlah userId : ', rating['userId'].nunique())
print('Jumlah blog_id : ', rating['blog_id'].nunique())
print('Jumlah data rating : ', len(rating))

"""Dari hasil code diatas terdapat user unik sebanyak 5001 data. Sedangkan blog unik sebanyak 9706. Secara keseluruhan, data pada dataset tersebut dengan total 200,140."""

blog.info()

"""Pada dataset 'blog' terdapat 8 kolom, yaitu 'blog_id', 'autho_id', 'blog_title', 'blog_content', 'blog_link', 'blog_img', 'topic', dan 'scrape_time'. Namun yang akan digunakan pada proses modelling hanya 3 kolom saja, yaitu kolom 'blog_id', 'blog_title', dan 'topic'. Sehingga pada proses ini hanya 3 kolom saja yang di*explore*."""

print('Jumlah blog_id : ', blog['blog_id'].nunique())
print('Jumlah topic : ', blog['topic'].nunique())
print('topic Blog: ', blog['topic'].unique())

"""Hasil code diatas ditemukan bahwa jumlah blog yaitu 10467, topic dengan 23.

### **Visualisasi Data**

Pada bagian ini, akan dilakukan beberapa visualisasi data untuk memahami dataset secara mendalam
"""

topics = blog.groupby('topic')['blog_id'].count().sort_values(ascending=False).head(10)
topics

"""Code diatas akan menampilkan 10 topik terbanyak yang ada pada blog. Agar lebih mudah membandingkannya, dilakukan visualisasi data"""

plt.figure(figsize=(10,5))

sns.barplot(y=topics.index, x=topics.values)
plt.title('10 Topik Teratas')
plt.xlabel('Total Blog')
plt.ylabel('Topik')
plt.show()

"""Menggunkan hasil visualisasi diatas, dapat dipahami lebih mudah bagaimana perbandingan setiap topik pada blog"""

blog_count_by_user = rating.groupby('userId')['blog_id'].count().sort_values(ascending=False).head(5)
blog_count_by_user

"""Code diatas digunakan untuk menampilkan 5 user yang paling sering memberikan rating pada suatu blog. Dapat dilihat yaitu user dengan ID 3619 memberikan rating kepada 374 blog."""

plt.figure(figsize=(10,5))

sns.barplot(x=blog_count_by_user.index, y=blog_count_by_user.values)
plt.title('5 User Terbanyak Memberi Rating')
plt.xlabel('ID User')
plt.ylabel('Total Blog')
plt.show()

"""Dengan hasil visualisasi tersebut, kita dapat mengetahui bagaimana perbandingan setiap user yang memberikan rating"""

average_ratings_by_blog = rating.groupby('blog_id')['ratings'].mean().sort_values(ascending=False).head(10)
print(average_ratings_by_blog)

"""Code diatas digunakan untuk memberikan hasil 10 ID Blog dengan rata-rata tertinggi."""

plt.figure(figsize=(10,5))

sns.barplot(x=average_ratings_by_blog.index, y=average_ratings_by_blog.values)
plt.title('10 Blog Dengan Rata-rata Rating Tertinggi')
plt.xlabel('ID Blog')
plt.ylabel('Rata-Rata Rating')
plt.show()

"""Digunakan visualiasi barplot untuk memahami data agar lebih mudah

# **DATA PREPROCESSING**

Pada proses data preprocessing ini, tidak terlalu banyak yang dilakukan. Proses yang dilakukan hanya penggabungan dataset 'rating' dan 'blog' saja.
"""

all_blog = pd.merge(rating, blog, on='blog_id', how='left')
all_blog.head()

"""# **DATA PREPARATION**

## **Asessing Data**

Pertama kali dilakukan adalah pengecekan missing value. Tujuan dari proses ini untuk mengetahui berapa jumlah data yang bernilai null.
"""

all_blog.isnull().sum()

"""Dari hasil code diatas dapat disimpulkan bahwa dataset 'all_blog' tidak memiliki missing value.

Selanjutnya dilakukan pengecekan duplikasi data
"""

all_blog.duplicated().sum()

"""Dari code diats, tidak ditemukan adanya duplikasi data.

## **Cleaning Data**

Pada dataset 'all_blog', tidak dibutuhkan beberapa kolom. Untuk mempermudah proses memahami data, hanya diambil kolom 'blog_id', 'blog_title', dan 'topic'.
"""

all_blog.drop(columns=['author_id', 'blog_content', 'blog_link', 'blog_img', 'scrape_time'], axis=1, inplace=True)
all_blog.head()

"""Pada kolom 'blog_title', terdapat judul yang memiliki emoji. Sehingga perlu dilakukan penghapusan emoji tersebut."""

def remove_dash(topic):
    return topic.replace('-', '')

demoji.download_codes()
all_blog['blog_title'] = all_blog['blog_title'].apply(lambda x: demoji.replace(x, ''))
# all_blog['topic'] = all_blog['topic'].apply(remove_dash)
all_blog

"""Selanjutnya dilihat topic yang ada pada dataset 'all_blog'"""

all_blog['topic'].unique()

"""Dari code diatas, tidak ada keanehan pada data di kolom 'topic' tersebut."""

preparation_df = all_blog
preparation_df.head()

"""Selanjutnya dataset 'all_blog' ditampung kedalam satu dataset yang baru. Dataset tersebut diberi nama 'preparation_df'

Untuk memastikan dataset 'preparation_df' benar-benar telah bersih, maka dijalankan code berikut.
"""

preparation_df.isnull().sum()

"""Kemudian dataset 'preparation_df' tersebut kita sorting berdasarkan nilai 'blog_valu'nya."""

preparation_df = preparation_df.sort_values('blog_id', ascending=True)
preparation_df

"""Selanjutnya untuk proses pemodelan, maka dihapus data duplikat menggunakan fungsi 'drop_duplicates()'."""

preparation_df = preparation_df.drop_duplicates('blog_id')
preparation_df

"""Tahap selanjutnya melakukan konversi dataseries menjadi list menggunakan fungsi tolist()."""

blog_id = preparation_df['blog_id'].tolist()
blog_name = preparation_df['blog_title'].tolist()
blog_topic = preparation_df['topic'].tolist()

print(len(blog_id))
print(len(blog_name))
print(len(blog_topic))

"""Tahap terakhir pada proses preparation adalah membuat dictionary untuk menentukan pasangan key-value pada data blog_id, blog_name, dan blog_topic"""

blog_new = pd.DataFrame({
    'id':blog_id,
    'blog_title':blog_name,
    'topic':blog_topic
})

blog_new

"""# **MODEL DEVELOPMENT (CONTENT BASED FILTERING)**

Selanjutnya kita pindahkan dataset 'blog' yang mengandung informasi mengenai blog kedalam satu variabel baru. Disini diberi nama 'data'
"""

data = blog_new
data.sample(5)

"""Selanjutnya digunakan TfidfVectorizer() untuk melakukan perhitungan idf pada data 'topic'. Pada hasil code berikut dilihat bahwa semua kalimat akan dipecah perkata. Tujuan ini untuk melihat bagaimana kekuatan setiap kata terhadap judul blog yang akan dianalisis"""

tf = TfidfVectorizer()

tf.fit(data['topic'])

tf.get_feature_names_out()

"""Setelah itu dilakukan fit dan transformasi kedalam bentuk matriks"""

tfidf_matrix = tf.fit_transform(data['topic'])

tfidf_matrix.shape

"""Dari hasil yang didapatkan, matriks yang dimiliki berukuran (9706,23). 9706 merupakan ukuran data dan 23 merupakan matrik kategori masukan.

Untuk menghasilkan tf-idf dalam bentuk matriks, maka digunakan fungsi todense()
"""

tfidf_matrix.todense()

"""Kemudian untuk melihat matriks idf untuk beberapa blog (blog_name) dan kategori topic (topic), maka dilakukan pembuatan dataframe yang baru."""

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=data.blog_title
).sample(22, axis=1).sample(10, axis=0)

"""Output dari matriks tersebut menampilkan angka antara 0.0 sampai 1.0. Nilai matriks 0.0 menunjukkan ketidakmiripan terhadap suatu topic. Sebaliknya, nilai matriks yang hampir mendekati atau sama dengan 1.0 menunjukkan kemiripan judul blog tersebut terhadap suatu topic.

Selanjutnya akan dihitung cosine_similiarity nya untuk menghitung derajat kesamaan (similiarity degree) antar blog dengan teknik cosine similiarity.
"""

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim

"""Kemudian untuk dijalankan code berikut untuk melihat kesamaan setiap blog dengan menampilkan nama blog dalam 5 sampel kolom (axis=1) dan 10 sampel baris (axis=0)."""

cosine_sim_df = pd.DataFrame(cosine_sim, index=data['blog_title'], columns=data['blog_title'])
print('Shape:', cosine_sim_df.shape)

cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""Tahap ini dilakukan pembuatan fungsi blog_recommendations yang akan dipanggil ketika ingin melihat rekomendasi. Terdapat beberapa parameter yaitu
- blog_title : berisi judul blog
- similiarity_data : Dataframe mengenai similiarity yang telah didefinisikan sebelumnya (cosine_sim_df).
items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan. Dalam hal ini adalah 'blog_title' dan 'topic'.
- k : Banyak rekomendasi yang akan diberikan.
"""

def blog_recommendations(blog_title, similarity_data=cosine_sim_df, items=data[['blog_title', 'topic']], k=5):
    index = similarity_data.loc[:,blog_title].to_numpy().argpartition(
        range(-1, -k, -1))

    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    closest = closest.drop(blog_title, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)

"""Selanjutnya untuk menemukan rekomendasi blog yang mirip dengan judul blog 'Why Every Organisation Needs Cloud Security Policies in the New Era', maka dijalankan code berikut untuk melihat topic sebenarnya."""

data[data.blog_title.eq('Why Every Organisation Needs Cloud Security Policies in the New Era')]

"""Setelah itu, untuk menguji coba hasilnya, maka dipanggil fungsi blog_recommendations untuk melihat rekomendasi yang ditampilkan"""

blog_recommendations('Why Every Organisation Needs Cloud Security Policies in the New Era')

"""Sistem berhasil memberikan 5 rekomendasi dengan judul blog. Kelima judul blog tersebut memiliki topic 'cloudcomputing'. Hal ini serupa dengan topic sebenarnya pada judul 'Why Every Organisation Needs Cloud Security Policies in the New Era'.

# **MODEL DEVELOPMENT (COLLABORATIVE FILTERING)**

Proses pertama yang dilakukan adalah, membuat satu variabel baru untuk menampung dataset rating yang telah di load sebelumnya.
"""

df = rating
df

"""Selanjutnya dilakukan proses penyandian (encode) firut 'userId', dan 'blog_id' kedalam indeks integer."""

user_ids = df['userId'].unique().tolist()
print('list userID: ', user_ids)

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded userId : ', user_to_user_encoded)

user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke userId: ', user_encoded_to_user)

"""Dapat dilihat bahwa nilai-nilai id yang ada pada 'userId' berubah seperti yang tampil

Begitu juga dengan 'blog_id'. Nilai pada kolom ini akan diencode atau disandikan
"""

blog_ids = df['blog_id'].unique().tolist()

blog_to_blog_encoded = {x: i for i, x in enumerate(blog_ids)}

blog_encoded_to_blog = {i: x for i, x in enumerate(blog_ids)}

"""Kemudian kita hitung jumlah dari user dan blog. Begitu juga dengan nilai minimal rating dan maksimal rating"""

num_users = len(user_to_user_encoded)
print(num_users)

num_blog = len(blog_encoded_to_blog)
print(num_blog)

min_rating = min(df['ratings'])

max_rating = max(df['ratings'])

print('Number of User: {}, Number of Resto: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_blog, min_rating, max_rating
))

"""Didapatkan bahwa jumlah user adalah 5001, jumlah blog 9706. Sedangkan rating minimal itu 0.5 dan maksimal 5.0

Selanjutnya nilai yang telah di encode, dimasukkan kedalam dataframe dengan kolom baru. Jadi terdapat 2 kolom baru yaitu 'user' dan 'blog'.
"""

df['user'] = df['userId'].map(user_to_user_encoded)

df['blog'] = df['blog_id'].map(blog_to_blog_encoded)

df.head()

"""Untuk membagi data training dan testing secara acak, maka dilakukan pengacakan menggunakan pengambilan sampel."""

df = df.sample(frac=1, random_state=42)
df

"""Kemduian buat 2 variabel baru. Variabel x akan menampung nilai 'user' dan 'blog'. Sedangkan variabel y akan menampung nilai 'ratings'. Perbandingan data training dan testing yang dibuat adalah 80:20."""

x = df[['user', 'blog']].values

y = df['ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)

from sklearn.model_selection import train_test_split
x = df[['userId', 'blog_id']].values
y = df['ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=123)

x_train.shape

"""Selanjutnya dibuat satu class baru dengan nama RecommenderNet dengna keras. Nantinya untuk merekomendasikan blog, akan dipanggil kelas tersebut."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class RecommenderNet(tf.keras.Model):

  def __init__(self, num_users, num_resto, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_blog = num_blog
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1)
    self.resto_embedding = layers.Embedding(
        num_blog,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.resto_bias = layers.Embedding(num_blog, 1)

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0])
    user_bias = self.user_bias(inputs[:, 0])
    resto_vector = self.resto_embedding(inputs[:, 1])
    resto_bias = self.resto_bias(inputs[:, 1])

    dot_user_resto = tf.tensordot(user_vector, resto_vector, 2)

    x = dot_user_resto + user_bias + resto_bias

    return tf.nn.sigmoid(x)

"""Selanjutnya kita panggil kelas tersebut dan mengcompile terhadap model. Disini digunakan optimizer SGD dengan learning_rate 0.01 dan momentum 0.9"""

model = RecommenderNet(num_users, num_blog, 50)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.8),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

"""Proses training pun dilakukan. Pada proses ini dilakukan pelatihan model dengan jumlah epoch adalah 10."""

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 10,
    epochs = 30,
    validation_data = (x_val, y_val)
)

"""Selanjutnya untuk mengevaluasi metrik root_mean_square yang dilatih, maka diplot hasilnya meggunakan library pyplot."""

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""Pada grafik diatas dapat dilihat bahwa nilai RMSE dominan turun setiap epoch yang dijalankan. Namun jika diperhatikan makapada epoch akhir-akhir, nilai test RMSE nya mulai naik kembali.

Selanjutnya kita cek apakah model tersebut telah memberikan rekomendasi yang blog yang sesuai.
"""

blog_df = blog_new
df = rating

user_id = df.userId.sample(1).iloc[0]
blog_read_by_user = df[df.userId == user_id]

blog_not_read = blog_df[~blog_df['id'].isin(blog_read_by_user.blog_id.values)]['id']
blog_not_read = list(
    set(blog_not_read)
    .intersection(set(blog_to_blog_encoded.keys()))
)

blog_not_read = [[blog_to_blog_encoded.get(x)] for x in blog_not_read]
user_encoder = user_to_user_encoded.get(user_id)
user_blog_array = np.hstack(
    ([[user_encoder]] * len(blog_not_read), blog_not_read)
)

"""Kemudian kita menggunakan model.predict untuk memperoleh rekomendasi yang diberikan"""

ratings = model.predict(user_blog_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_blog_ids = [
    blog_encoded_to_blog.get(blog_not_read[x][0]) for x in top_ratings_indices
]

print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('Resto with high ratings from user')
print('----' * 8)

top_blog_user = (
    blog_read_by_user.sort_values(
        by = 'ratings',
        ascending=False
    )
    .head(5)
    .blog_id.values
)

blog_df_rows = blog_df[blog_df['id'].isin(top_blog_user)]
for row in blog_df_rows.itertuples():
    print(row.blog_title, ':', row.topic)

print('----' * 8)
print('Top 10 Blog recommendation')
print('----' * 8)

recommended_blog = blog_df[blog_df['id'].isin(recommended_blog_ids)]
for row in recommended_blog.itertuples():
    print(row.blog_title, ':', row.topic)

"""Dapat dilihat bahwa rekomendasi yang diberikan oleh siste. User tersebut memberi rating tertinggi kepada blog yang memiliki topic 'data-analysis'. Sehingga pada bagian rekomendasi beberapa blog yang direkomendasikan bertopik data analysis dan ai. Tidak hanya itu jika dilihat secara spesifik, judul dari blog yang direkomendasikan juga masih memiliki hubungan dengan data."""