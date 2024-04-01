# Laporan Proyek Machine Learning - M. Iqbal Purba

## Project Overview

Konsumsi akses berita telah mengalami perubahan dari model tradisional seperti koran, majalah dan media cetak lainnya menjadi akses ribuan berita dari berbagai sumber di internet [[1]](http://jurnal.unmuhjember.ac.id/index.php/SENSEI17/article/view/1064/854). Proses pencarian manual untuk menemukan artikel yang sesuai di internet bisa menjadi tugas yang melelahkan dan memakan waktu bagi pengguna. Salah satu cara untuk membuat akses pengunjung lebih lama dari sebuah website adalah membuat sistem rekomendasi [[1]](http://jurnal.unmuhjember.ac.id/index.php/SENSEI17/article/view/1064/854). Layanan yang terbaik yang diberikan salah satunya yaitu membantu pengguna untuk menawarkan pilihan berita yang disesuaikan dengan minat dan ketertarikan pembaca [[2]](http://jurnalti.polinema.ac.id/index.php/SIAP/article/view/559/194).

Sistem rekomndasi merupakan perangkat lunak (_software_) dan kumpulan teknik yang dapat memberikan saran untuk item yang sangat sesuai dengan preferensi suatu _user_ tertentu [[3]](http://ejurnal.itats.ac.id/snestik/article/view/2635/2258). Beberapa algoritma yang kerap dipakai dalam sistem rekomendasi mencakup _Collaborative Filtering, Content-Based Filtering,_ serta _Hybrid Filtering_ [[4]](https://ejournal.itn.ac.id/index.php/jati/article/view/7073/4533). Hasil rekomendasi yang dihasilkan oleh sistem berperan dalam membantu pengguna menemukan barang atau materi yang serasi dengan keinginan mereka, serta meningkatkan keseluruhan pengalaman pengguna [[4]](https://ejournal.itn.ac.id/index.php/jati/article/view/7073/4533).

Pengembangan sistem rekomendasi memainkan peran penting dalam pengalaman pengguna di platform berita online. Hal ini dikarenakan sistem rekomendasi dapat membantu mengatasi fenomena "filter bubble", dimana pengguna hanya terpapar pada sudut pandang atau topik tertentu. Dengan memperkenalkan berbagai sumber dan topik yang mungkin tidak ditemukan oleh pengguna secara alami, sistem rekomendasi dapat membantu menghadirkan perspektif yang lebih luas. Tidak hanya itu, sistem rekomendasi juga membantu menghemat waktu pengguna dengan menyajikan konten yang paling mungkin diminati sehingga pengguna akan lebih banyak menghabiskan waktu membaca.

Sistem rekomendasi juga memungkinkan personalisasi konten berita berdasarkan preferensi dan perilaku pengguna. Personalisasi ini tidak hanya memastikan relevansi konten, tetap juga meningkatkan peluang pengguna untuk menemukan berita yang benar-benar menarik bagi mereka. Hal ini tentunya dapat meningkatkan pengalaman pengguna. Pengguna juga dapat lebih lama tinggal dan menjelajahi _platform_ jika mereka menemukan konten yang relevan dan menarik. 

Berdasarkan penelitian yang dilakukan oleh [[5]](https://ejournal.unesa.ac.id/index.php/jinacs/article/view/43953/37432), didapatkan bahwa metode _Content-Based Filtering_ dan algoritma _Random Forest Regression_ dapat membantu memudahkan para pengguna mendapatkan penyedia jasa pernikahan sesuai dengan _budget_ yang dimiliki dan lokasi yang diinginkan. Penelitian lain juga mendapatkan kesimpulan bahwa metode _Item-based Collaborative Filtering_ dapat digunakan dalam memberikan rekomendasi produk kepada pelanggan, _Cosine Similarity_ memberikan hasil yang baik dalam menghitung tingkat kemiripan antar produk dan metode ini menjadi referensi dan acuan perusahaan dalam memberikan rekomendasi kepada pelanggan [[6]](https://jidt.org/jidt/article/view/151/86). Selain itu, pada penelitian [[1]] didapatkan bahwa metode KNN dapat digunakan sebagai sistem rekomendasi artikel berita berdasarkan _tagline_ yang telah ditentukan dengan uji keakuratan berdasarkan nilai error sebesar 14%.

## Business Understanding

### Problem Statements

Pernyataan masalah:
- Bagaimana membuat sistem rekomendasi yang dapat memberikan hasil berdasarkan _content_ dan kebiasaan _user_?
- Bagaimana perbandingan rekomendasi yang diberikan oleh metode _Content Based Filtering_ dan _Collaborative Filtering_?

### Goals

Tujuan :
- Untuk membuat sistem rekomendasi yang dapat memberikan hasil berdasarkan _content_ dan kebiasaan _user_
- Untuk membandingkan rekomendasi yang diberikan oleh _Content Based Filtering_ dan _Collaborative Filtering_

### Solution statements
  Untuk mendapatkan tujuan diatas maka dilakukan pembangunan model menggunakan metode _Content Based Filtering_ dan _Collaborative Filtering_. Pada proses pembangunan model menggunakan  _Content Based Filtering_, dilakukan perhitungan IDF untuk menemukan bagaimana kekuatan setiap kata terhadap judul _blog_ menggunakan TfIdfVectorizer(). Kemudian dilihat bagaimana hubungan antara judul _blog_ dengan _topic_ yang ada pada _blog_ dan hubungan antara judul _blog_. Selanjutnya dihitung _similiarity_nya menggunakan *cosine_similiarity*. Sedangkan untuk _Collaborative Filtering_ digunakan proses encode terlebih dahulu  dan melatih modelnya menggunakan metode _deep learning_.

  Perbandingan antara _Content-Based Filtering_ (CBF) dan _Collaborative Filtering_ (CF) merupakan langkah krusial dalam pengembangan sistem rekomendasi. Kedua metode ini memiliki pendekatan yang berbeda dalam memberikan rekomendasi kepada pengguna, dan memahami kelebihan serta kelemahan masing-masing sangat penting untuk memilih metode yang sesuai dengan kebutuhan dan karakteristik platform. 
  
  Pada CBF, mempertimbangkan karakteristik atau konten dari item yang direkomendasikan, misalkan topik, genre untuk membuat rekomendasi. Sedangkan pada CF menggunakan informasi dari perilaku atau preferensi  pengguna untuk membuat rekomendasi. Selain itu, Kedua metode ini juga berbeda dalam data yang dibutuhkan. CBF membutuhkan informasi yang cukup tentang karakteristik atau fitur dari item yang direkomendasikan. Berbeda dengan CBF, CF justru membutuhkan data interaksi pengguna, seperti riwayat rating atau preferensi untuk menentukan kesamaan antar pengguna atau item. _Content-Based Filtering_ umumnya lebih mudah untuk diimplementasikan dan lebih skalabel karena tidak terlalu bergantung pada data interaksi pengguna. Sedangkan _Collaborative Filtering_ membutuhkan perhitungan kompleks dan dapat menghadapi tantangan skalabilitas dengan pertumbuhan jumlah pengguna dan item. Sehingga hal tersebut menjadi alasan penting untuk membandingkan kedua metode tersebut.

  Untuk membandingkan kedua model tersebut, maka dihitung metrik evaluasinya. Untuk model pertama digunakan metrik evaluasi _precision_ untuk melihat bagaimana rekomendasi yang diberikan. Sedangkan untuk model kedua digunakan metrik evaluasi _Root Mean Squared Error (RMSE)_. 


## Data Understanding
Dataset yang digunakan pada proyek ini diambil melalui _Kaggle_, dapat diunduh pada link berikut [*Blog Recommendation*](https://www.kaggle.com/datasets/yakshshah/blog-recommendation-data). Pada link tersebut terdapat 3 dataset ('Blog Ratings', 'Medium Blog Data', 'Author Data'), namun yang hanya digunakan adalah 2 dataset saja ('rating', 'blog'). Pada dataset *'Blog Ratings'* memiliki 3 kolom, yaitu *blog_id, userId*, dan *ratings*. Masing-masing kolom memiliki 200.140 data. Pada dataset tersebut tidak ditemukan adanya _missing value_ maupun _duplicated value_. Jumlah _userId_ yang unik adalah 5001 data, sedangkan jumlah *blog_id* yang unik adalah 9706 data. Pada dataset *'Medium Blog Data'*, terdapat 8 kolom yaitu *blog_id, author_id, blog_title, blog_content, blog_link, blog_img, topic* dan *scrape_time* yang masing-masing terdiri dari 10.467 data. Tidak ditemukan adanya _missing value_ maupun _duplicated value_. Jumlah *blog_id* yang unik adalah 10467, sedangkan jumlah *topic* yang unik adalah 23. Namun, pada dataset *'Medium Blog Data'* hanya digunakan 3 kolom saja yaitu *blog_id, blog_title*, dan *topic*.

Pada 2 dataset tersebut, terdapat 2 data yang menarik untuk dijadikan sebagai parameter untuk memberikan rekomendasi, yaitu _rating_ dan _topic_. Pada dataset *'Blog Rating'* terdapat kolom bernama _rating_ yang memiliki nilai antara 0.5 - 5.0. Total data _rating_ yang dapat diolah pada dataset tersebut sebesar 200.140 data. Pada dataset *'Medium Blog Data'* terdapat kolom _topic_ yang berisi 23 jenis topik. Untuk melihat detail dari topik tersebut, dapat dilihat pada kolom 1.

Variabel-variabel pada *Blog Ratings* dataset adalah sebagai berikut:
- *blog_id* : nilai unik yang menjadi identitas dari blog. Nilai pada kolom *blog_id* ini akan menjadi *foreign key* untuk menggabungkan dataset *Blog Ratings* dengan *Medium Blog Data*.
- *userId* : Nilai unik yang menjadi identitas dari _user_ yang memberikan penilaian (_rating_) pada suatu blog. Nilai pada kolom ini memiliki tipe integer.
- *ratings* : Merupakan penilaian (*rating*) yang diberikan oleh _user_. Kolom ini berisi nilai antara 0.5 - 5.0. Tipe nilai dari kolom ini adalah _float_.

Variabel-variabel pada *Medium Blog Data* dataset adalah sebagai berikut:
- *blog_id* : Nilai unik yang menjadi identitas dari suatu _blog_. Nilai pada kolom *blog_id* ini akan menjadi *foreign key* untuk menggabungkan dataset *Blog Ratings* dengan *Medium Blog Data*.
- *author_id* : Nilai unik yang menjadi identitas dari penulis suatu _blog_. 
- *blog_title* : Menampilkan judul dari setiap blog. Setiap judul _blog_ akan memiliki 1 *blog_id* saja.
- *blog_content* : Menampilkan deskripsi singkat yang ada pada suatu _blog_.
- *blog_link* : Menampilkan link yang dapat diakses oleh _user_ untuk membaca blog tersebut.
- *blog_img* : Menampilkan link dari _cover_ suatu blog.
- *topic* : Menampilkan topik yang dibahas pada suatu *blog*.
- *scrape_time* : Menampilkan waktu ketika data di koleksi.


Untuk memahami data tersebut dilakukan proses _Exploratory Data Analysis (EDA)_. Proses EDA yang dilakukan adalah _Univariate Analysis_. Tujuan dari _Univariate Analysis_ adalah untuk mengetahui total dan nilai dari setiap dataset. Dataset *'Blog Ratings'* di*load* terlebih dahulu kedalam proyek, kemudian ditampung dalam variabel bernama 'rating'. Setelah itu, dilakukan _Univariate Analysis_ untuk mengetahui total data yang ada. Dari hasil analisis ditemukan bahwa terdapat 200.140 data dari setiap kolomnya. Sedangkan jumlah unik dari userId dan blog_id secara beruturut-turut adalah 5001 dan 9706. Selanjutnya dilakukan _Univariate Analysis_ terhadap dataset *Medium Blog Data*. Pertama yang dilakukan adalah me*load* dataset tersebut dan disimpan dalam variabel 'blog'. Dari hasil analisis didapatkan bahwa terdapat 10467 data di setiap kolom. Jumlah unik *blog_id* dan _topic_ secara berturut-turut adalah 10.467 dan 23 data. Sedangkan nilai apa saja yang ada di _topic_ dapat dilihat pada kolom 1 berikut : 


| Komponen       | Nilai                                                                                                                                                                                                                                                                                                                                               |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jumlah blog_id | 10467                                                                                                                                                                                                                                                                                                                                               |
| Jumlah topic   | 23                                                                                                                                                                                                                                                                                                                                                  |
| Topic Blog     | 'ai' 'image-processing' 'Cryptocurrency' 'data-science' 'dev-ops',, 'security', 'android', 'cloud-computing', 'nlp', 'cloud-services', 'flutter', 'web3', 'cybersecurity', 'information-security', 'blockchain', 'machine-learning', 'deep-learning', 'data-analysis', 'backend', 'backend-development', 'app-development', 'Software-Development'. |

Kolom 1 : Hasil _Univariate Analysis_ dataset *Medium Blog Data*

Pada gambar 1 diatas, ditemukan bahwa terdapat 23 topik dari blog-blog yang ada, diantaranya yaitu 'ai', 'image_processing', 'Cryptocurrency', 'data-science' dan yang lainnya.

Selain itu, dilakukan juga visualisasi data terhadap dataset yang ada. Pada dataset 'blog' dilakukan visualisasi terhadap kolom _topic_ untuk menemukan top 10 topik paling banyak. Untuk hasil visualisasinya dapat dilihat pada gambar 1 berikut :

![image](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/a165a754-37b7-4fe2-af7b-52b9fe4064ab) 
Gambar 1 : Top 10 Topik Paling Banyak

Pada gambar 1 tersebut disimpulkan bahwa blog yang memiliki topik terbanyak diduduki oleh topik 'ai', yaitu lebih dari 700 blog. Kemudian diikuti oleh topik 'blockchain' dengan total blog lebih dari 600 blog. Pada posisi 10 diduduki oleh topik 'nlp' dengan total blog lebih dari 453 blog.

Tidak hanya itu, dilakukan juga proses visualisasi terhadap dataset 'rating'. Pada dataset ini akan divisualisasikan 5 _user_ yang paling banyak memberikan _rating_ kepada suatu blog. Untuk hasil visualisasinya dapat dilihat pada gambar 2 berikut.

![image](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/eafd2acf-386c-46a2-9a23-2f7511f77eb5)
Gambar 2 : Top 5 *User* yang paling banyak memberikan *rating*

Dari gambar 2 diatas didapatkan *insight* bahwa _user_ dengan ID 3619 memberikan rating kepada lebih dari 350 blog, diikuti dengan ID 3882 dengan hampir 300 blog. Pada urutan kelima diduduki oleh _user_ dengan ID 4453.

Masih dengan dataset yang sama, divisualisasikan 10 blog yang mendapatkan rata-rata 'rating' tertinggi. Untuk hasilnya dapat dilihat pada gambar 3 berikut.

![image](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/7f47540f-51e2-4898-aec6-ff02b73bd1a4)
Gambar 3 : Top 10 Blog yang memiliki rata-rata *rating* tertinggi

Dari gambar 3 ditemukan bahwa blog dengan ID 1342, 3246, 1296, 4463, 1246, dan 6705 mendapatkan rata-rata rating tertinggi yaitu 5.00. Sedangkan 4 lainnya yang dapat dilihat pada gambar diatas, mendapatkan rata-rata *rating* 4.8 dan terendah mendapatkan 4.7.

## Data Preparation
Sebelum dilakukan proses preparation, dilakukan proses preprocessing data. Proses preprocessing yang dilakukan adalah menggabungkan kedua dataset tersebut menggunakan nilai *'blog_id'*, kemudian disimpan dataset yang telah digabungkan tersebut kedalam 'all_blog'. 

Kemudian dicek apakah ada data yang *missing value* atau *duplicated value* dari dataset yang telah digabungkan tersebut. Ternyata dari hasil pengecekan, tidak ditemukan adanya *missing value* atau *duplicated value*. Alasan dilakukannya pengecekan *missing value* dan *duplicated value* dari dataset adalah untuk meningkatkan kualitas dataset. Jika data yang hilang atau terdapat duplikasi data maka terdapat kemungkinan model yang dihasilkan tidak akan akurat atau bahkan proses pembangunan model akan mengalami error.

Setelah itu, dilakukan proses *cleaning data*. Kolom yang hanya dibutuhkan pada saat pembangunan model nanti adalah *'blog_id', 'userId', 'ratings', 'blog_title',* dan *'topic'*. Alasan penghapusan beberapa kolom lainnya karena yang akan menjadi faktor memberikan rekomendasi hanya berdasarkan *'ratings'* dan *'topic'* saja. Sehingga kolom yang lain tidak akan memberikan pegaruh yang berarti. Namun untuk memberikan rekomendasi yang lebih spesifik dan lebih personalisasi, pengerjaan proyek selanjutnya disarankan untuk menggunakan beberapa parameter yang lainnya, seperti konten dari blog atau penulis dari blog tersebut. 

Setelah itu, jika dilihat nilai pada kolom *'blog_title'*, maka terdapat _emoticon/emoji_, sehingga perlu dilakukan penghapusan _emoticon/emoji_ tersebut.Tujuan dari pembersihan *emoticon/emoji* adalah untuk meningkatkan kualitas dataset karena algoritma _Machine Learning_ tidak mengenali emoticon. Pertama yang dilakukan adalah meng*install* demoji kemudian meng*import library* demoji. Selanjutnya dijalakan kode berikut : 
```
demoji.download_codes()
all_blog['blog_title'] = all_blog['blog_title'].apply(lambda x: demoji.replace(x, ''))
```

Kode tersebut jika dijalankan maka akan  menghilangkan emoji dari nilai yang ada di kolom *'blog_title'*. Kemudian, dilakukan peng*copy*an dataset yang telah dilakukan *assessing* dan *cleaning data* kedalam variabel bernama *'preparation_df'*.

Dataset *'preparation_df'* tersebut selanjutnya di*sorting* (diurutkan) berdasarkan nilai 'blog_id'. Setelah itu, dilakukan penghapusan *duplicated value* yang ada. Jika dilihat, maka panjang masing-masing data yang ada di kolom *'blog_id', 'blog_title',* dan *'blog_topic'* adalah 9706. Kemudian ketiga kolom tersebut dimasukkan kedalam 1 *DataFrame* baru dengan nama *'blog_new'*. Nantinya, dataset *'blog_new'* tersebut yang akan dijadikan sebagai dataset pembangunan model.

Selanjutnya proses pembagian dataset menjadi data training dan validation pun dilakukan. Pertama perlu membuat 2 variabel yang akan menampung data. Variabel x digunakan untuk menampung data dari fitur *user* dan *blog*. Sedangkan variabel y digunakan untuk menampung data *rating* yang telah dinormalisasi. Selanjutnya dilakukan pembagian dataset dengan rasio 80:20. Maksudnya adalah 80% digunakan untuk pelatihan, dan 20% digunakan untuk validasi.

## Modeling

Pada proyek ini diberikan dua solusi untuk memecahkan masalah yaitu metode _Content Based Filtering_ dan _Collaborative Filtering_.

### _Content Based Filtering_
Sistem rekomendasi berbasis konten (Content-based Recommendation System) menggunakan ketersediaan konten (sering juga disebut dengan fitur, atribut atau karakteristik) sebuah item sebagai basis dalam pemberian rekomendasi. Secara umum, metode content-based filtering mempunyai 2 teknik umum dalam membuat rekomendasi yaitu heuristic-based dan model-based. Cosine similarity, Boolean query, teknik TF-IDF (term frequency-invers document frequency) dan Clustering termasuk dalam golongan heuristic-based sedangkan yang masuk dalam golongan model-based adalah teknik Bayesian classifier & Clustering, Decision Tree dan Artificial Neural Network. Metode content-based filtering, item direkomendasikan berdasarkan perbandingan antara profil item dan profil pengguna. 

Cara kerja _Content Based Filtering_ adalah menggunakan fitur atau karakteristik dari item yang direkomendasikan. Misalnya pada _platform_ berita, fitur bisa berupa kata kunci, topik, atau genre artikel. Pengguna diberikan rekomendasi berdasarkan kesesuaian antara preferensi pengguna dengan fitur fitur item.  Contoh nyatanya adalah, _user_ A telah membaca artikel terkait _Natural Language Processing_ yang ditulis oleh B. Sedangkan B juga menulis artikel terkait _Backend Development_. Jadi artikel tentang _Backend Development_ dapat direkomendasikan kepada A.

Kelebihan _Content Based Filtering_ adalah :
1. Memberikan rekomendasi yang personal dan dapat menjelaskan alasan dibaik rekomendasi tersebut.
2. Interpretabilitas tinggi, maksudnya adalah model _Content Based Filtering_ dapat lebih mudah diinterpretasikan karena rekomendasi didasarkan pada karakteristik konten yang jelas.
3. Tidak bergantung pada data eksternal, maksudnya adalah tidak memerlukan data historis pengguna lain. Model hanya bergantung pada informasi fitur dari item yang direkomendasikan.

Kekurangan _Content Based Filtering_ adalah :
1. Kurangnya kemampuan untuk merekomendasikan item yang sangat berbeda dari yang telah disukai oleh pengguna.
2. Keterbatasan dalam mencari kesamaan. _Content Based Filtering_ kurang efektif dalam menemukan item atau konten yang memiliki kesamaan konsep atau tematik tetapi berbeda dalam representasi fitur.
3. Tidak secara dinamis menanggapi evolusi preferensi pengguna. Model harus diperbarui secara manual ketika preferensi berubah.

Pada proyek ini, digunakan nilai pada kolom *'topic'* untuk memberikan rekomendasi kepada _user_. Penggunaan TF-IDF (*TfIdfVectorizer()*) digunakan untuk melihat kata-kata penting yang ada pada kolom *'topic'*. Nilai-nilai pada kolom tersebut akan dipecah untuk menemukan kekuatan setiap kata terhadap judul blog tersebut. Nilai dari hasil analisis ini adalah antara 0-1. Semakin besar (mendekati 1) nilainya, maka semakin kuat hubungannya, begitu juga sebaliknya. 

Setelah itu, dilakukan perhitungan kedekatan atau _similiarity_ menggunakan *cosine_similiarity*. Adapun parameter yang digunakan untuk menghitung *cosine_similiarity* adalah *blog_title* dan *topic*. Untuk megetahui hasil dari pemodelan ini, maka dibuat uji coba. Diambil secara acak judul blog yang akan digunakan. Dijalankan code berikut :
```
data[data.blog_title.eq('Why Every Organisation Needs Cloud Security Policies in the New Era')]
```

Judul blog tersebut memiliki topik *cloud-computing*. Oleh karena itu seharusnya blog yang direkomendasikan adalah memiliki topik yang sama, yaitu *cloud-computing*. Berikut hasilnya yang disajikan dalam tabel 2 berikut :

| blog_title                                                                                   | topic           |
| -------------------------------------------------------------------------------------------- | --------------- |
| Building an ETL Pipeline on the Cloud for Big Data Processing- “Formula-1 racing case study” | cloud-computing |
| Common AWS interview questions for Freshers: AWS Articles                                    | cloud-computing |
| Coffee Bytes with The Cloud — Tony Stark Cut                                                 | cloud-computing |
| How To Choose A Cloud Provider For Your Business                                             | cloud-computing |
| How to Create an Amazon AWS Cognito User Pool                                                | cloud-computing |

Tabel 2 : *Top 5 Recommendations*

Dari tabel 2 terlihat bahwa ada 5 blog yang direkomendasikan berdasarkan judul yang dimasukkan sebelumnya. Dapat diilihat bahwa kelima blog tersebut memiliki topic yang sama dengan topic blog *input*an yaitu *cloud-computing*. Sehingga model tersebut telah berhasil dibuat.


### _Collaborative Filtering_

_Collaborative filtering_ adalah teknik dalam sistem rekomendasi yang populer digunakan saat ini. Banyak penelitian yang membahas tentang teknik ini karena beberapa keunggulannya seperti: menghasilkan serendipity(tak terduga) item, sesuai trend market, mudah diimplementasikan dan memumgkinkan diterapkan pada beberapa domain _(book, movies, music, dll)_. Cara kerja teknik ini adalah dengan memanfaatkan data pada komunitas dengan cara mencari kemiripan atar pengguna, yaitu mengasumsikan bahwa pengguna yang memiliki preferensi serupa di masa lalu cenderung memiliki preferensi yang sama di masa depan. 

Secara umum, _Collaborative Filtering_ dibagi menjadi 2 yaitu *user-user Collaborative Filtering* dan *item-item Collaborative Filtering*. Pada *user-user Collaborative Filtering*, metode ini membandingkan preferensi pengguna dengan pengguna lain. Misalnya jika pengguna A dan B memiliki preferensi serupa, maka item yang disukai oleh pengguna B dan belum dilihat oleh A dapat direkomendasikan kepada A. Metode ini memanfaatkan matriks preferensi pengguna untuk menghitung kemiripan antar pengguna. Sedangkan *item-item Collaborative Filtering* akan membandingkan kesamaan antar item berdasarkan preferensi pengguna. Misalnya, jika pengguna A menyukai item X, dan item Y memiliki kesamaan dengan item X, maka Y dapat direkomendasikan kepada A. Metode ini memanfaatkan matriks preferensi item untuk menghitung kemiripan antar item.

Kelebihan dari _Collaborative Filtering_ adalah : 
1. Mampu menemukan rekomendasi yang tidak mungkin ditemukan oleh pengguna karena berdasarkan pada pola perilaku pengguna yang mirip.
2. Lebih efektif dalam menangani data yang sparsitas, karena mengandalkan informasi kolaboratif dari pengguna lain.
3. TIdak memerlukan pengetahuan yang mendalam tentang konten atau item. Hanya membutuhkan data historis pengguna untuk memberikan rekomendasi.

Kekurangan dari _Collaborative Filtering_ adalah :
1. Memerlukan data historis pengguna yang berpotensi menimbulkan kekhawatiran privasi terutama jika informasi pengguna yang spesifik diperlukan untuk memberikan rekomendasi yang akurat.
2. Kesulitan menangani item baru yang belum mendapatkan cukup data pengguna, karena memerlukan interaksi pengguna yang signifikan untuk memberikan rekomendasi yang akurat.

Pada proyek ini yang pertama kali dilakukan adalah proses _encode_ pada fitur *'userId'*, dan *'blog_id'*. Kemudian hasil _encode_ dimasukkan kedaam _DataFrame_. Kemudian masuk kedalam tahap pembagian data training dan validation. Proses yang dilakukan adalah mengacak dataset terlebih dahulu. Kemudian membuat variabel X untuk menampung data _'user'_, dan *'blog'*. Sedangkan variabel y menampung data *'ratings'*. Perbandingan data training dengan validation adalah 80%:20%. Setelah itu dilakukan proses pembuatan *class* dengan nama *RecommenderNet* dengan *keras.* Nantinya untuk merekomendasikan blog, akan dipanggil *class* tersebut. Pada proses *compile*, digunakan *BinaryCrossentropy()*. *Optimizer* yang digunakan adalah SGD dengan *learning rate* sebesar 0.01 dan *momentum* 0.8. Sedangkan metrik yang digunakan adalah RMSE. Untuk uji coba, maka akan dicari rekomendasi berdasarkan _user_ dengan *userID* 3968. Tabel 3 akan menampilkan 5 blog dengan *rating* tertinggi yang diberikan user.

| blog_title                                                                        | Topic         |
| --------------------------------------------------------------------------------- | ------------- |
| Essential Excel Functions for Data Analysis and Manipulation                      | data-analysis |
| Building Your Own Langchain Agents and Tools with LLMs: A Step-by-Step Tutorial   | data-analysis |
| ChatGPT in Excel Data Analysis                                                    | data-analysis |
| CHOOSING AN ANALYTICAL TOOL: SAS, R, OR PYTHON                                    | data-analysis |
| Introduction to Machine Learning for Data Analysts and for Non-Tech Professionals | data-analysis |

Tabel 3 : *Top 5 Rating From User*

Dari tabel 3 tersebut dapat dilihat bahwa, _user_ dengan *userId* 3968 memberikan _rating_ tertinggi kepada blog yang memiliki topik *data-analysis*. Selanjutnya pada tabel 4 akan ditampilkan rekomendasi dari sistem.

| blog_title                                                                                            | topic                |
| ----------------------------------------------------------------------------------------------------- | -------------------- |
| Infrastructure as Code using Terraform                                                                | dev-ops              |
| The rise and rise of ‘data-driven’ businesses                                                         | cybersecurity        |
| InfoSecSherpa’s News Roundup for Monday, March 20, 2023                                               | information-security |
| Top 5 Data Management Trends Taking the Finance Industry by Storm in 2023.                            | security             |
| Machine Learning 101: A Beginner’s Guide                                                              | ai                   |
| Starting Your Data Analytics Journey: A Comprehensive Guide for college students and recent graduates | data-analysis        |
| What to do if you lose your software engineer job.                                                    | backend-development  |
| Beyond the Bot: The Human Skills AI Can’t Replace                                                     | ai                   |
| Instagram OSINT & Hacking — Phishing at its best.                                                     | cybersecurity        |
| Deploying a MERN Stack on AWS                                                                         | dev-ops              |

Tabel 4 : *Top 10 Recommendations*


Dari tabel 4 tersebut, hasil rekomendasi ditemukan bahwa hanya ada 1 yang memiiki topik yang sama. Namun, judul rekomendasi yang lain masih memiliki kesamaan dengan judul yang diberikan rating tertinggi. Seperti yang kita ketahui bahwa, *data analysis* menggunakan teknologi *Artificial Intelligence* sehingga topik tersebut pun ikut direkomendasikan. Tidak hanya itu, jika diperhatikan lebih lanjut, judul blog yang direkomendasikan juga masih berhubungan dengan data.

## Evaluation

Evaluasi dilakukan berdasarkan metrik evaluasi yang digunakan. Pada metode _Content Based Filtering_ maka digunakan metrik evaluasi *precision*. Metrik tersebut mengukur rasio prediksi positif yang benar dari semua prediksi positif. *Precision* memberikan informasi tentang seberapa sering model *machine learning* membuat prediksi positif yang benar. Nilai precision berkisar antara 0 dan 1. Gambar 4 menampilkan formula dari metrik *precision*.

![ss8](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/0f8bc751-cb25-46d7-9af3-3460dbe3a22d)
Gambar 4 : Formula metrik *precision*

Pada gambar 4 tersebut dihitung metrik precision dari model yang telah dibangun menggunakan metode _Content Based Filtering_. Pada bagian *modelling* diatas, dapat dilihat bahwa dilakukan uji coba menggunakan judul blog *Why Every Organisation Needs Cloud Security Policies in the New Era* dengan *topic cloud-computing*. Jika ingin menghitung *precision* dari model tersebut, maka dapat dimasukkan pada rumus di gambar 4. 

Jumlah prediksi benar : 5
Prediksi benar : 5
Preidksi Salah : 0

Sehingga perhitungannya dapat dilihat pada gambar 5 berikut : 

![ss9](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/f18fb8ca-dca7-49b8-8c9e-127e0a6f684b)
Gambar 5 : Perhitungan metrik *Precision*

Dari gambar 5 tersebut dapat dilihat bahwa nilai *precision* dari model yang dibangun adalah 1. Nilai ini tergolong sempurna untuk menentukan ketepatan hasil rekomendasi.

Selanjutnya pada metode yang menggunakan *Collaborative Filtering*, digunakan metrik evaluasi *Root Mean Squared Error* (RMSE). Untuk formula dari metrik ini dapat dilihat pada gambar 6 berikut.

![rmse](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/cbeb581f-6f23-43cc-ac9d-eff67cde0525)
Gambar 6 : Formula metrik *Root Mean Squared Error* (RMSE)

Dari gambar 6 tersebut, dapat disimpulkan bahwa *Root Mean Square Error* (RMSE) adalah ukuran yang sering digunakan untuk mengukur kesalahan prediksi model. Intinya, ini memberitahu tentang distribusi residu (kesalahan prediksi). RMSE yang lebih rendah menunjukkan kecocokan data yang lebih baik. Pada model menggunakan *Collaborative Filtering* didapatkan rmse pada saat pelatihan model. Gambar 7 menampilkan hasilnya sebagai berikut.

![ss11](https://github.com/iqbalpurba26/aset_laporan/assets/100006429/0e25e3a1-7775-4bb4-b5ca-fccecaf9ed89)
Gambar 7 : Grafik RMSE model *Collaborative Filtering* 

Pada gambar 7 tersebut dapat dilihat bahwa nilai RMSE dari model dominan turun pada data training maupun validation. Namun, pada data validation nilai RMSE nya sedikit naik pada epoch akhir. Secara umum, hasil dari _plot_ grafik tersebut telah menunjukkan hasil yang _goodfit_. Namun hal ini masih belum cukup baik jika diinginkan rekomendasi yang lebih valid. Untuk mengatasi hal tersebut dapat dilakukan dengan menambahkan fitur yang dapat dijadikan sebagai parameter untuk menentukan rekomendasi. Tidak hanya itu, jika dilihat pada garis _validation_, disaat beberapa epoch terakhir nilai RMSE nya menjadi naik. Hal ini dapat diatasi dengan menggunakan fungsi _callback_ untuk menghentikan proses _epoch_ jika sudah mencapai kondisi yang diinginkan.

## Referensi 
[1] 	S. Abraham and Y. D. Rahayu, "SISTEM REKOMENDASI ARTIKEL BERITA MENGGUNAKAN METODE K-NEAREST NEIGHBOR BERBASIS WEBSITE," in Prosiding SENSEI, Jember, 2017. 

[2] 	R. Rismanto, M. Mentari and R. Nurwidhi, "Rekomendasi Artikel Terkait Pada Berita Online Menggunakan Teknik Text Mining," in Seminar Informatika Aplikatif Polinema, Malang, 2019. 

[3] 	I. H. Putri, S. M. K. H. Nurakhmadyavi and E. E. Wahyudi, "Literature Review: Sistem Rekomendasiuntuk Buku dan Film," in Seminat Nasional Teknik Elektro, Sistem Informasi, dan Teknik Informatika, Surabaya, 2022. 

[4] 	L. I. Sidora and N. H. Harani, "SISTEM REKOMENDASI MUSIK SPOTIFY MENGGUNAKAN KNN DAN ALGORITMA GENETIKA," Jurnal Mahasiswa Teknik Informatika, vol. 7, no. 4, pp. 2585-2591, 2023. 

[5] 	D. A. Pratiwi and A. Qoiriah, "Sistem Rekomendasi Wedding Organizer Menggunakan Metode Content-Based Filtering Dengan Algoritma Random Forest Regression," Journal of Informatics and Computer Science, vol. 3, no. 3, pp. 231-239, 2022. 

[6] 	D. Theodorus, S. Defit and G. W. Nurchayo, "Machine Learning Rekomendasi Produk dalam Penjualan Menggunakan Metode Item-Based Collaborative Filtering," Jurnal Informasi dan Teknologi, vol. 3, no. 4, pp. 202-208, 2021. 
