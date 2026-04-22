"""
Presentation / Slide module — Perkuliahan Data Mining
Pemateri: Rahmad Ade Akbar, S.Pd., M.Pd
"""

import streamlit as st


def render_presentation():
    """Render interactive presentation slides."""

    slides = [
        # ─── 10 Slide Pengenalan di Awal ───────────────────────
        _slide_cover,
        _slide_about_lecturer,
        _slide_outline,
        _slide_what_is_data_mining,
        _slide_dm_vs_ml,
        _slide_model_overview,
        _slide_model_knn,
        _slide_model_svm,
        _slide_model_random_forest,
        _slide_model_logistic_regression,
        _slide_model_neural_network,
        _slide_model_kmedoids_kmeans,
        _slide_model_ensemble_methods,
        _slide_why_dt_nb,
        _slide_platforms_tools,
        # ─── Slide Materi Inti ─────────────────────────────────
        _slide_what_is_ml,
        _slide_supervised_learning,
        _slide_classification_intro,
        _slide_decision_tree_concept,
        _slide_decision_tree_how,
        _slide_decision_tree_splitting,
        _slide_decision_tree_pros_cons,
        _slide_naive_bayes_concept,
        _slide_naive_bayes_theorem,
        _slide_naive_bayes_types,
        _slide_naive_bayes_pros_cons,
        _slide_evaluation_metrics,
        _slide_confusion_matrix,
        _slide_workflow,
        _slide_comparison,
        _slide_real_world,
        _slide_demo_intro,
        _slide_thank_you,
    ]

    if "slide_idx" not in st.session_state:
        st.session_state.slide_idx = 0

    total = len(slides)
    idx = st.session_state.slide_idx

    col_prev, col_info, col_next = st.columns([1, 3, 1])

    with col_prev:
        if st.button("Sebelumnya", disabled=(idx == 0), use_container_width=True):
            st.session_state.slide_idx -= 1
            st.rerun()

    with col_info:
        st.markdown(
            f"<div style='text-align:center; padding:8px; color:#3a86ff; font-weight:600;'>"
            f"Slide {idx + 1} / {total}</div>",
            unsafe_allow_html=True,
        )

    with col_next:
        if st.button("Selanjutnya", disabled=(idx == total - 1), use_container_width=True):
            st.session_state.slide_idx += 1
            st.rerun()

    st.progress((idx + 1) / total)
    st.markdown("---")

    slides[idx]()

    with st.expander("Daftar Semua Slide"):
        slide_names = [
            "1. Cover",
            "2. Tentang Pemateri",
            "3. Outline Materi",
            "4. Apa itu Data Mining?",
            "5. Data Mining vs Machine Learning",
            "6. Overview Model-Model Klasifikasi",
            "7. K-Nearest Neighbors (KNN)",
            "8. Support Vector Machine (SVM)",
            "9. Random Forest",
            "10. Logistic Regression",
            "11. Neural Network / Deep Learning",
            "12. K-Means & K-Medoids Clustering",
            "13. Ensemble Methods",
            "14. Mengapa Decision Tree & Naive Bayes?",
            "15. Platform & Tools Data Mining",
            "16. Apa itu Machine Learning?",
            "17. Supervised Learning",
            "18. Pengenalan Klasifikasi",
            "19. Konsep Decision Tree",
            "20. Cara Kerja Decision Tree",
            "21. Splitting Criteria",
            "22. Pro & Kontra Decision Tree",
            "23. Konsep Naive Bayes",
            "24. Teorema Bayes",
            "25. Tipe-tipe Naive Bayes",
            "26. Pro & Kontra Naive Bayes",
            "27. Metrik Evaluasi",
            "28. Confusion Matrix",
            "29. Workflow ML",
            "30. Perbandingan DT vs NB",
            "31. Aplikasi Dunia Nyata",
            "32. Intro Demo",
            "33. Terima Kasih",
        ]
        selected = st.selectbox("Pilih slide:", slide_names, index=idx)
        new_idx = slide_names.index(selected)
        if new_idx != idx:
            st.session_state.slide_idx = new_idx
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# SLIDE BARU — PENGENALAN (10 slide di awal)
# ═══════════════════════════════════════════════════════════════

def _slide_cover():
    st.markdown(
        """
        <div style="text-align:center; padding: 40px 20px;">
            <h1 style="font-size:2.8rem; color:#1a1a2e; margin-bottom:0.5rem;">
                Perkuliahan Data Mining
            </h1>
            <h2 style="font-size:1.8rem; color:#3a86ff; margin-bottom:1rem;">
                Membangun Model Prediksi
            </h2>
            <h3 style="font-size:1.4rem; color:#555; font-weight:400;">
                Klasifikasi Data dengan<br>
                <strong>Decision Tree</strong> & <strong>Naive Bayes</strong>
            </h3>
            <hr style="width:200px; margin:30px auto; border-color:#3a86ff;">
            <p style="font-size:1.1rem; color:#888;">
                Pemateri: <strong>Rahmad Ade Akbar, S.Pd., M.Pd</strong><br>
                Mata Kuliah Data Mining
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_about_lecturer():
    st.markdown("## Tentang Pemateri")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(
            "https://media.licdn.com/dms/image/v2/D5603AQHx7g0mSgcwag/profile-displayphoto-scale_200_200/B56ZtyfZ_lHkAY-/0/1767152404946?e=1778112000&v=beta&t=sunasOBlvz3ypxwFV3aZW4Qh8ayVMYFX1gdpx8tnC8M",
            width=200,
            caption="Rahmad Ade Akbar, S.Pd., M.Pd",
        )
        st.markdown(
            """
            **Full Stack Developer | System Analyst | Trainer | Master of Education**

            [github.com/rahmadadeakbar](https://github.com/rahmadadeakbar)
            """
        )

    with col2:
        st.markdown(
            """
            ### Rahmad Ade Akbar, S.Pd., M.Pd
            """
        )

        st.markdown(
            """
            | Posisi | Instansi |
            |--------|----------|
            | **Senior Developer** | paperlesshospital.id 
            | **Lecturer** | UIN Ar-Raniry, Banda Aceh
            | **Founder** | Karir Beasiswa Hub & Raa Technomedia
            | **Senior Full Stack Developer** | Institute of Education and Social Research 
            | **Full Stack Developer** | Irwan Abdullah Scholar Foundation Journals 
            """
        )

        st.markdown(
            """
            **Bidang Keahlian:**
            - Full Stack Developer
            - Data Mining & Machine Learning
            - System Analysis
            - Higher Education Teaching

            ---
            *"Data is the new oil, but mining it wisely is the real skill."*
            """
        )


def _slide_outline():
    st.markdown("## Outline Materi")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Bagian I — Pengenalan
            1. **Apa itu Data Mining?**
               - Definisi, proses, dan hubungannya dengan ML
            2. **Overview Model-Model Klasifikasi**
               - KNN, SVM, Random Forest, Logistic Regression, dll.
            3. **Platform & Tools Data Mining**
               - Software dan library yang bisa digunakan

            ### Bagian II — Teori Inti
            4. **Decision Tree**
               - Konsep, cara kerja, splitting criteria
            5. **Naive Bayes**
               - Teorema Bayes, asumsi, tipe-tipe
            """
        )

    with col2:
        st.markdown(
            """
            ### Bagian III — Praktik
            6. **Evaluasi Model**
               - Accuracy, Precision, Recall, F1-Score
            7. **Hands-on Demo**
               - Eksplorasi data (EDA)
               - Preprocessing
               - Training & evaluasi
               - Perbandingan model
            8. **Prediksi Interaktif**
               - Live prediction dengan model

            ### Bagian IV — Evaluasi
            9. **Soal Praktikum**
               - Testing pemahaman peserta
            """
        )


def _slide_what_is_data_mining():
    st.markdown("## Apa itu Data Mining?")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            <strong>Definisi:</strong><br>
            Data Mining adalah proses <strong>menemukan pola, korelasi, dan anomali</strong>
            dari kumpulan data besar menggunakan teknik statistik, matematika, dan
            komputasi untuk <strong>mengubah data menjadi informasi yang berguna</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    st.markdown(
        """
        ### Proses KDD (Knowledge Discovery in Databases)

        ```
        Data  -->  Selection  -->  Preprocessing  -->  Transformation  -->  Data Mining  -->  Evaluation  -->  Knowledge
        ```

        ### Tugas Utama Data Mining

        | Tugas | Deskripsi | Contoh |
        |-------|-----------|--------|
        | **Klasifikasi** | Memprediksi kategori/kelas | Spam vs Bukan Spam |
        | **Clustering** | Mengelompokkan data serupa | Segmentasi pelanggan |
        | **Asosiasi** | Menemukan hubungan antar item | Market basket analysis |
        | **Regresi** | Memprediksi nilai kontinu | Prediksi harga rumah |
        | **Deteksi Anomali** | Menemukan data tidak biasa | Deteksi fraud |
        """
    )


def _slide_dm_vs_ml():
    st.markdown("## Data Mining vs Machine Learning")
    st.markdown("---")

    st.markdown(
        """
        | Aspek | Data Mining | Machine Learning |
        |-------|------------|-----------------|
        | **Tujuan** | Menemukan pola dari data yang ada | Belajar dari data untuk prediksi masa depan |
        | **Fokus** | Eksplorasi & penemuan pengetahuan | Membangun model prediktif |
        | **Data** | Biasanya data historis statis | Data bisa terus bertambah (streaming) |
        | **Proses** | Bagian dari KDD | Komponen teknis dari Data Mining |
        | **Output** | Pola, aturan, insight | Model terlatih |
        | **Interaksi** | Banyak melibatkan analis manusia | Lebih otomatis (self-learning) |
        """
    )

    st.markdown(
        """
        <div class="info-box">
            <strong>Hubungan:</strong> Machine Learning adalah salah satu
            <strong>teknik/alat utama</strong> yang digunakan dalam proses Data Mining.
            Dalam perkuliahan ini kita menggunakan teknik ML untuk menyelesaikan
            tugas-tugas Data Mining.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_model_overview():
    st.markdown("## Overview Model-Model Klasifikasi")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            Sebelum membahas Decision Tree & Naive Bayes secara mendalam, mari kita kenali
            <strong>berbagai algoritma klasifikasi</strong> yang populer di dunia Data Mining.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        | No | Algoritma | Tipe | Kompleksitas | Interpretasi |
        |----|-----------|------|-------------|-------------|
        | 1 | **K-Nearest Neighbors (KNN)** | Instance-based | Rendah | Mudah |
        | 2 | **Support Vector Machine (SVM)** | Margin-based | Tinggi | Sulit |
        | 3 | **Random Forest** | Ensemble (Tree) | Sedang | Sedang |
        | 4 | **Logistic Regression** | Probabilistik | Rendah | Mudah |
        | 5 | **Neural Network** | Connectionist | Sangat Tinggi | Sangat Sulit |
        | 6 | **Decision Tree** | Rule-based | Rendah | Sangat Mudah |
        | 7 | **Naive Bayes** | Probabilistik | Rendah | Mudah |
        | 8 | **Gradient Boosting (XGBoost)** | Ensemble (Boosting) | Tinggi | Sedang |

        > Slide berikutnya akan membahas masing-masing model secara singkat.
        """
    )


def _slide_model_knn():
    st.markdown("## K-Nearest Neighbors (KNN)")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Konsep
            - Klasifikasi berdasarkan **K tetangga terdekat**
            - Menggunakan **jarak** (Euclidean, Manhattan, dll.)
            - **Lazy learner** — tidak ada fase training eksplisit

            ### Cara Kerja
            1. Tentukan nilai K
            2. Hitung jarak data baru ke semua data training
            3. Ambil K data terdekat
            4. Voting mayoritas → kelas prediksi
            """
        )

    with col2:
        st.markdown(
            """
            ### Kelebihan
            - Sederhana & mudah dipahami
            - Tidak ada asumsi distribusi data
            - Efektif untuk data non-linear

            ### Kekurangan
            - Lambat untuk dataset besar (hitung jarak ke semua)
            - Sensitif terhadap **skala fitur** (perlu normalisasi)
            - Sensitif terhadap **dimensi tinggi** (curse of dimensionality)
            - Pemilihan K sangat berpengaruh
            """
        )

    st.markdown(
        """
        <div class="warning-box">
            <strong>Tips:</strong> Selalu lakukan <strong>normalisasi/standarisasi</strong> data
            sebelum menggunakan KNN, dan gunakan <strong>cross-validation</strong> untuk memilih K terbaik.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_model_svm():
    st.markdown("## Support Vector Machine (SVM)")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Konsep
            - Mencari **hyperplane terbaik** yang memisahkan kelas
            - Memaksimalkan **margin** antara kelas
            - **Support vectors** = data paling dekat ke boundary

            ### Kernel Trick
            - **Linear**: Data terpisah secara linear
            - **RBF (Gaussian)**: Data non-linear
            - **Polynomial**: Batas keputusan polinomial
            """
        )

    with col2:
        st.markdown(
            """
            ### Kelebihan
            - Efektif untuk **high-dimensional** data
            - Tahan terhadap overfitting (margin maximization)
            - Fleksibel dengan berbagai kernel

            ### Kekurangan
            - Lambat untuk dataset **sangat besar**
            - Sensitif terhadap **pemilihan kernel** & parameter C
            - Sulit diinterpretasikan (black box)
            - Tidak cocok untuk data **noisy**
            """
        )

    st.markdown(
        """
        <div class="info-box">
            <strong>Best for:</strong> Text classification, image recognition,
            dan data dengan dimensi tinggi namun sampel terbatas.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_model_random_forest():
    st.markdown("## Random Forest")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Konsep
            - **Ensemble** dari banyak Decision Tree
            - Setiap tree dilatih pada **subset acak** dari data (Bagging)
            - Setiap split menggunakan **subset acak fitur**
            - Prediksi akhir = **voting mayoritas** dari semua tree

            ### Cara Kerja
            1. Buat N bootstrap samples dari data
            2. Latih Decision Tree pada setiap sample
            3. Setiap tree memilih fitur acak saat splitting
            4. Gabungkan prediksi semua tree (voting)
            """
        )

    with col2:
        st.markdown(
            """
            ### Kelebihan
            - **Mengurangi overfitting** (vs single Decision Tree)
            - Akurasi tinggi & stabil
            - Menangani missing values dengan baik
            - **Feature importance** otomatis
            - Robust terhadap outlier

            ### Kekurangan
            - Lebih **lambat** dari single Decision Tree
            - **Sulit diinterpretasikan** (banyak tree)
            - Membutuhkan lebih banyak **memori**
            - Bisa bias pada fitur dengan banyak level
            """
        )


def _slide_model_logistic_regression():
    st.markdown("## Logistic Regression")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Konsep
            - Memodelkan **probabilitas** kelas menggunakan fungsi logistik (sigmoid)
            - Output: probabilitas antara 0 dan 1
            - Threshold (biasanya 0.5) untuk klasifikasi

            ### Formula Sigmoid

            $$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$

            Di mana $z = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n$
            """
        )

    with col2:
        st.markdown(
            """
            ### Kelebihan
            - Sederhana & cepat
            - Output berupa **probabilitas** (interpretable)
            - Tidak rentan overfitting (dengan regularisasi)
            - Baik sebagai **baseline model**

            ### Kekurangan
            - Asumsi **hubungan linear** antara fitur dan log-odds
            - Tidak cocok untuk data **non-linear** kompleks
            - Sensitif terhadap **multikolinearitas**
            - Perlu **feature engineering** untuk performa optimal
            """
        )

    st.markdown(
        """
        <div class="info-box">
            <strong>Meskipun namanya "Regression"</strong>, Logistic Regression adalah
            algoritma <strong>klasifikasi</strong>. Nama "regression" merujuk pada teknik
            statistik yang digunakannya.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_model_neural_network():
    st.markdown("## Neural Network / Deep Learning")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Konsep
            - Terinspirasi dari **otak manusia** (neuron biologis)
            - Terdiri dari **layer**: Input → Hidden → Output
            - Setiap neuron memproses sinyal dan meneruskannya
            - **Deep Learning** = Neural Network dengan banyak hidden layers

            ### Arsitektur Umum
            ```
            Input Layer  →  Hidden Layer(s)  →  Output Layer
            [x1, x2, xn]    [weights, bias]     [class 0/1]
                              activation fn
            ```
            """
        )

    with col2:
        st.markdown(
            """
            ### Kelebihan
            - Sangat kuat untuk data **kompleks & non-linear**
            - Mampu **feature extraction otomatis**
            - State-of-the-art di banyak bidang (vision, NLP)
            - Fleksibel untuk berbagai jenis data

            ### Kekurangan
            - Butuh **data sangat banyak**
            - **Komputasi intensif** (GPU/TPU)
            - **Black box** — sulit diinterpretasikan
            - Rentan **overfitting** tanpa regularisasi
            - Banyak **hyperparameter** yang perlu di-tuning
            """
        )

    st.markdown(
        """
        <div class="warning-box">
            <strong>Catatan:</strong> Untuk dataset kecil-menengah seperti yang kita gunakan
            hari ini, model-model tradisional seperti Decision Tree & Naive Bayes sering
            memberikan hasil <strong>setara atau bahkan lebih baik</strong> dari Neural Network.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_model_kmedoids_kmeans():
    st.markdown("## K-Means & K-Medoids Clustering")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            Meskipun bukan algoritma klasifikasi (melainkan <strong>clustering/unsupervised</strong>),
            K-Means dan K-Medoids sering dibahas dalam Data Mining sebagai teknik pengelompokan data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### K-Means
            - Mengelompokkan data ke **K cluster**
            - **Centroid** = rata-rata data dalam cluster
            - Iteratif: assign → update centroid → repeat

            **Kelebihan:** Cepat, sederhana
            **Kekurangan:** Sensitif terhadap outlier, harus tentukan K
            """
        )

    with col2:
        st.markdown(
            """
            ### K-Medoids (PAM)
            - Mirip K-Means, tapi pusat cluster = **data point nyata** (medoid)
            - Lebih **robust terhadap outlier**
            - Bisa menggunakan berbagai **distance metric**

            **Kelebihan:** Tahan outlier, lebih fleksibel
            **Kekurangan:** Lebih lambat dari K-Means
            """
        )

    st.markdown(
        """
        ### Perbedaan dengan Klasifikasi

        | Clustering (Unsupervised) | Klasifikasi (Supervised) |
        |--------------------------|-------------------------|
        | **Tanpa label** — data mengelompokkan sendiri | **Ada label** — belajar dari contoh |
        | Menemukan **struktur** tersembunyi | Memprediksi **kelas** data baru |
        | Contoh: K-Means, K-Medoids, DBSCAN | Contoh: Decision Tree, Naive Bayes |
        """
    )


def _slide_model_ensemble_methods():
    st.markdown("## Ensemble Methods")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            <strong>Ensemble Methods</strong> menggabungkan beberapa model untuk menghasilkan
            prediksi yang <strong>lebih akurat dan stabil</strong> daripada model tunggal.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### Bagging
            - **Bootstrap Aggregating**
            - Latih model pada subset acak data
            - Voting mayoritas / rata-rata
            - Contoh: **Random Forest**

            *Mengurangi variance*
            """
        )

    with col2:
        st.markdown(
            """
            ### Boosting
            - Latih model **berurutan**
            - Setiap model fokus pada **kesalahan** model sebelumnya
            - Contoh: **AdaBoost, XGBoost, LightGBM**

            *Mengurangi bias*
            """
        )

    with col3:
        st.markdown(
            """
            ### Stacking
            - Gabungkan prediksi dari **model berbeda**
            - Gunakan **meta-model** untuk prediksi akhir
            - Contoh: DT + NB + SVM → LogReg

            *Memanfaatkan kekuatan masing-masing model*
            """
        )

    st.markdown(
        """
        <div class="success-box">
            <strong>Fun Fact:</strong> Hampir semua pemenang kompetisi Kaggle menggunakan
            <strong>ensemble methods</strong>, terutama <strong>XGBoost</strong> dan <strong>LightGBM</strong>!
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_why_dt_nb():
    st.markdown("## Mengapa Kita Pelajari Decision Tree & Naive Bayes?")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            Dari sekian banyak algoritma, mengapa kita fokus pada dua model ini?
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Alasan Akademik
            - **Fundamental** — Memahami konsep dasar klasifikasi
            - **Interpretable** — Mudah dipahami & dijelaskan
            - **Representatif** — Mewakili dua pendekatan berbeda:
              - Decision Tree: **Rule-based**
              - Naive Bayes: **Probabilistik**
            - **Dasar untuk model lanjutan** (Random Forest ← DT)
            """
        )

    with col2:
        st.markdown(
            """
            ### Alasan Praktis
            - **Cepat** — Training & prediksi sangat cepat
            - **Tidak butuh data banyak**
            - **Cocok sebagai baseline** sebelum model kompleks
            - **Mudah diimplementasikan** dengan Scikit-learn
            - **Banyak digunakan** di industri untuk kasus-kasus tertentu
            """
        )

    st.markdown(
        """
        <div class="success-box">
            <strong>Prinsip:</strong> Selalu mulai dari model sederhana. Jika hasilnya sudah baik,
            tidak perlu model kompleks. <em>"Keep it simple!"</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_platforms_tools():
    st.markdown("## Platform & Tools untuk Data Mining")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Bahasa Pemrograman & Library

            | Tool | Deskripsi |
            |------|-----------|
            | **Python + Scikit-learn** | Library ML paling populer, mudah digunakan |
            | **Python + Pandas** | Manipulasi & analisis data |
            | **Python + TensorFlow/PyTorch** | Deep Learning framework |
            | **R + caret/tidymodels** | Statistik & ML untuk akademisi |
            | **Julia** | Bahasa komputasi ilmiah cepat |

            ### IDE & Notebook

            | Tool | Deskripsi |
            |------|-----------|
            | **Jupyter Notebook** | Notebook interaktif untuk eksplorasi |
            | **Google Colab** | Jupyter di cloud, gratis GPU |
            | **VS Code + Extensions** | Editor kode dengan ML extensions |
            | **RStudio** | IDE khusus R |
            | **Kaggle Notebooks** | Notebook + dataset + kompetisi |
            """
        )

    with col2:
        st.markdown(
            """
            ### Platform GUI (Tanpa Coding)

            | Tool | Deskripsi |
            |------|-----------|
            | **WEKA** | Software DM open-source (Java), cocok akademik |
            | **RapidMiner** | Visual workflow drag-and-drop |
            | **KNIME** | Platform analitik visual open-source |
            | **Orange** | Data mining visual programming (Python-based) |
            | **IBM SPSS Modeler** | Enterprise data mining tool |

            ### Platform Cloud & Big Data

            | Tool | Deskripsi |
            |------|-----------|
            | **Google Cloud AutoML** | Automated ML tanpa coding |
            | **AWS SageMaker** | ML platform end-to-end |
            | **Azure ML Studio** | Visual ML drag-and-drop |
            | **Databricks** | Unified analytics platform |
            | **H2O.ai** | AutoML open-source |
            """
        )

    st.markdown(
        """
        <div class="success-box">
            <strong>Rekomendasi untuk Mahasiswa:</strong> Mulai dengan <strong>Python + Scikit-learn</strong>
            di <strong>Google Colab</strong> (gratis!), lalu eksplorasi <strong>WEKA</strong> atau
            <strong>Orange</strong> untuk pemahaman visual tanpa coding.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# SLIDE MATERI INTI (dari versi sebelumnya)
# ═══════════════════════════════════════════════════════════════

def _slide_what_is_ml():
    st.markdown("## Apa itu Machine Learning?")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            <strong>Definisi:</strong><br>
            Machine Learning adalah cabang dari Artificial Intelligence (AI) yang memungkinkan
            komputer untuk <strong>belajar dari data</strong> dan membuat keputusan atau prediksi
            <strong>tanpa diprogram secara eksplisit</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### Supervised Learning
            - Belajar dari data **berlabel**
            - Contoh: Klasifikasi, Regresi
            - *"Guru memberi contoh soal & jawaban"*
            """
        )

    with col2:
        st.markdown(
            """
            ### Unsupervised Learning
            - Belajar dari data **tanpa label**
            - Contoh: Clustering, Dimensionality Reduction
            - *"Siswa mengelompokkan sendiri"*
            """
        )

    with col3:
        st.markdown(
            """
            ### Reinforcement Learning
            - Belajar dari **reward/punishment**
            - Contoh: Game AI, Robotics
            - *"Belajar dari trial & error"*
            """
        )


def _slide_supervised_learning():
    st.markdown("## Supervised Learning")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            Supervised Learning menggunakan dataset yang memiliki <strong>input (fitur/X)</strong>
            dan <strong>output (label/Y)</strong> untuk melatih model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Klasifikasi
            Output berupa **kategori/kelas diskrit**

            | Input | Output |
            |-------|--------|
            | Email | Spam / Tidak Spam |
            | Gambar | Kucing / Anjing |
            | Data Pasien | Sakit / Sehat |
            """
        )

    with col2:
        st.markdown(
            """
            ### Regresi
            Output berupa **nilai kontinu**

            | Input | Output |
            |-------|--------|
            | Luas rumah | Harga (Rp) |
            | Pengalaman | Gaji (Rp) |
            | Suhu | Penjualan es |
            """
        )

    st.markdown(
        """
        <div class="warning-box">
            <strong>Fokus hari ini:</strong> Klasifikasi menggunakan Decision Tree & Naive Bayes
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_classification_intro():
    st.markdown("## Pengenalan Klasifikasi")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            <strong>Klasifikasi</strong> adalah proses memprediksi <strong>kelas/kategori</strong>
            dari suatu data berdasarkan atribut-atribut yang dimilikinya.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Contoh Kasus Nyata

        | Domain | Problem | Kelas |
        |--------|---------|-------|
        | Kesehatan | Diagnosis penyakit jantung | Sakit / Sehat |
        | Email | Deteksi spam | Spam / Bukan Spam |
        | Perbankan | Deteksi fraud | Fraud / Legitimate |
        | Pendidikan | Prediksi kelulusan | Lulus / Tidak Lulus |
        | E-commerce | Prediksi pembelian | Beli / Tidak Beli |

        ### Algoritma Klasifikasi Populer
        - Decision Tree
        - Naive Bayes
        - Neural Network / Deep Learning
        - K-Nearest Neighbors (KNN)
        - Support Vector Machine (SVM)
        - Random Forest
        """
    )


def _slide_decision_tree_concept():
    st.markdown("## Decision Tree — Konsep")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            <strong>Decision Tree</strong> adalah algoritma klasifikasi yang membuat keputusan
            dalam bentuk <strong>struktur pohon</strong>. Setiap node internal merepresentasikan
            sebuah <strong>tes pada atribut</strong>, cabang merepresentasikan <strong>hasil tes</strong>,
            dan daun merepresentasikan <strong>kelas/keputusan</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    st.markdown(
        """
        ### Analogi Sederhana

        ```
        Apakah cuaca cerah?
        |-- Ya --> Apakah angin kencang?
        |          |-- Ya  --> [Tidak bermain]
        |          +-- Tidak --> [Bermain]
        +-- Tidak --> Apakah hujan?
                   |-- Ya  --> [Tidak bermain]
                   +-- Tidak --> [Bermain]
        ```

        ### Komponen Decision Tree
        - **Root Node**: Node paling atas (pertanyaan pertama)
        - **Internal Node**: Node percabangan (pertanyaan lanjutan)
        - **Leaf Node**: Node akhir (keputusan/kelas)
        - **Branch**: Cabang yang menghubungkan node
        """
    )


def _slide_decision_tree_how():
    st.markdown("## Cara Kerja Decision Tree")
    st.markdown("---")

    st.markdown(
        """
        ### Proses Pembangunan Decision Tree

        1. **Pilih atribut terbaik** sebagai root node
           - Gunakan kriteria splitting (Information Gain / Gini Index)
        2. **Bagi data** berdasarkan nilai atribut terpilih
        3. **Ulangi proses** secara rekursif untuk setiap cabang
        4. **Hentikan** ketika:
           - Semua data di node sudah satu kelas (murni)
           - Tidak ada atribut tersisa
           - Mencapai kedalaman maksimum
        """
    )

    st.markdown(
        """
        ### Contoh: Prediksi Penyakit Jantung

        ```
        [Seluruh Data: 300 pasien]
                    |
            Chest Pain Type?
            /              \\
        Type 0-1          Type 2-3
        [180 pasien]      [120 pasien]
            |                  |
        Max Heart Rate?    Age > 55?
        /       \\          /       \\
      >150     <=150     Ya      Tidak
      [Sehat]  [Cek]    [Sakit]  [Cek]
        ```
        """
    )


def _slide_decision_tree_splitting():
    st.markdown("## Splitting Criteria")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Information Gain (ID3 / C4.5)

            Berbasis **Entropy** (ukuran ketidakteraturan):

            $$Entropy(S) = -\\sum_{i=1}^{c} p_i \\log_2(p_i)$$

            $$IG(S, A) = Entropy(S) - \\sum_{v} \\frac{|S_v|}{|S|} Entropy(S_v)$$

            - Entropy = 0 → Data **murni** (1 kelas)
            - Entropy = 1 → Data **sangat campuran**
            - Pilih atribut dengan **IG tertinggi**
            """
        )

    with col2:
        st.markdown(
            """
            ### Gini Index (CART)

            Mengukur **impurity** (ketidakmurnian):

            $$Gini(S) = 1 - \\sum_{i=1}^{c} p_i^2$$

            $$Gini_{split} = \\sum_{v} \\frac{|S_v|}{|S|} Gini(S_v)$$

            - Gini = 0 → Data **murni** (1 kelas)
            - Gini = 0.5 → Data **sangat campuran**
            - Pilih atribut dengan **Gini terendah**
            """
        )

    st.markdown(
        """
        <div class="warning-box">
            <strong>Catatan:</strong> Scikit-learn menggunakan <strong>Gini Index</strong> sebagai default
            untuk DecisionTreeClassifier, tetapi mendukung juga <strong>Entropy</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_decision_tree_pros_cons():
    st.markdown("## Decision Tree — Kelebihan & Kekurangan")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Kelebihan
            - **Mudah dipahami** & divisualisasikan
            - **Tidak perlu normalisasi** data
            - Bisa menangani data **numerik & kategorikal**
            - Dapat menangkap **hubungan non-linear**
            - **Feature importance** otomatis
            - Cepat dalam prediksi
            """
        )

    with col2:
        st.markdown(
            """
            ### Kekurangan
            - Rentan terhadap **overfitting**
            - **Tidak stabil** — perubahan kecil pada data bisa mengubah tree
            - Cenderung bias pada fitur dengan banyak level
            - Bisa menjadi sangat **kompleks**
            - Tidak optimal untuk hubungan yang **sangat linear**
            """
        )

    st.markdown(
        """
        <div class="success-box">
            <strong>Solusi Overfitting:</strong> Gunakan <em>pruning</em>
            (max_depth, min_samples_leaf) atau gunakan <strong>Random Forest</strong> (ensemble).
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_naive_bayes_concept():
    st.markdown("## Naive Bayes — Konsep")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            <strong>Naive Bayes</strong> adalah algoritma klasifikasi berbasis
            <strong>Teorema Bayes</strong> dengan asumsi bahwa setiap fitur
            <strong>independen satu sama lain</strong> (asumsi "naive").
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Intuisi

        **Bayangkan seorang dokter** mendiagnosis pasien:
        - Dokter sudah tahu **probabilitas umum** penyakit jantung (prior)
        - Pasien datang dengan **gejala tertentu** (evidence)
        - Dokter **mengupdate keyakinannya** berdasarkan gejala (posterior)

        ### Konsep Kunci
        - **Prior**: Probabilitas awal sebelum melihat data ($P(C)$)
        - **Likelihood**: Probabilitas fitur mengingat kelas ($P(X|C)$)
        - **Evidence**: Probabilitas fitur ($P(X)$)
        - **Posterior**: Probabilitas kelas mengingat fitur ($P(C|X)$) — **yang kita cari!**
        """
    )


def _slide_naive_bayes_theorem():
    st.markdown("## Teorema Bayes")
    st.markdown("---")

    st.markdown(
        """
        ### Formula Teorema Bayes

        $$P(C|X) = \\frac{P(X|C) \\cdot P(C)}{P(X)}$$

        | Simbol | Nama | Penjelasan |
        |--------|------|------------|
        | $P(C|X)$ | **Posterior** | Probabilitas kelas C diberikan fitur X |
        | $P(X|C)$ | **Likelihood** | Probabilitas fitur X diberikan kelas C |
        | $P(C)$ | **Prior** | Probabilitas awal kelas C |
        | $P(X)$ | **Evidence** | Probabilitas fitur X (normalisasi) |

        ### Dengan Asumsi "Naive" (Independen)

        $$P(X|C) = P(x_1|C) \\cdot P(x_2|C) \\cdot ... \\cdot P(x_n|C) = \\prod_{i=1}^{n} P(x_i|C)$$

        ### Keputusan Klasifikasi

        $$\\hat{y} = \\arg\\max_{c} P(C=c) \\prod_{i=1}^{n} P(x_i|C=c)$$
        """
    )


def _slide_naive_bayes_types():
    st.markdown("## Tipe-Tipe Naive Bayes")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### Gaussian NB
            - Fitur berdistribusi **normal**
            - Cocok untuk data **kontinu**
            - Estimasi mean dan std per kelas

            $$P(x_i|C) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} e^{-\\frac{(x_i - \\mu)^2}{2\\sigma^2}}$$

            **Paling umum digunakan**
            """
        )

    with col2:
        st.markdown(
            """
            ### Multinomial NB
            - Fitur berupa **frekuensi/count**
            - Cocok untuk **text classification**
            - Menggunakan distribusi multinomial

            **Aplikasi:**
            - Spam detection
            - Sentiment analysis
            - Document classification
            """
        )

    with col3:
        st.markdown(
            """
            ### Bernoulli NB
            - Fitur berupa **biner (0/1)**
            - Cocok untuk data **boolean**
            - Menggunakan distribusi Bernoulli

            **Aplikasi:**
            - Ada/tidak ada kata
            - Feature presence
            - Binary attributes
            """
        )


def _slide_naive_bayes_pros_cons():
    st.markdown("## Naive Bayes — Kelebihan & Kekurangan")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Kelebihan
            - **Sangat cepat** (training & prediksi)
            - Bekerja baik dengan **dataset kecil**
            - Efektif untuk **high-dimensional data**
            - **Tidak sensitif** terhadap fitur tidak relevan
            - Mudah diimplementasikan
            - Baik untuk **baseline model**
            """
        )

    with col2:
        st.markdown(
            """
            ### Kekurangan
            - Asumsi independensi **jarang terpenuhi**
            - Estimasi probabilitas bisa **kurang akurat**
            - Tidak menangkap **interaksi antar fitur**
            - Sensitif terhadap **distribusi data**
            - "Zero frequency problem" — perlu **Laplace smoothing**
            """
        )

    st.markdown(
        """
        <div class="success-box">
            <strong>Tips:</strong> Meskipun asumsi independensi jarang terpenuhi sepenuhnya,
            Naive Bayes sering memberikan <strong>hasil yang surprisingly baik</strong> dalam praktik!
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_evaluation_metrics():
    st.markdown("## Metrik Evaluasi Model")
    st.markdown("---")

    st.markdown(
        """
        | Metrik | Formula | Penjelasan |
        |--------|---------|------------|
        | **Accuracy** | $\\frac{TP + TN}{TP + TN + FP + FN}$ | Proporsi prediksi yang benar secara keseluruhan |
        | **Precision** | $\\frac{TP}{TP + FP}$ | Dari yang diprediksi positif, berapa yang benar? |
        | **Recall** | $\\frac{TP}{TP + FN}$ | Dari yang sebenarnya positif, berapa yang terdeteksi? |
        | **F1-Score** | $2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$ | Harmonik mean dari Precision & Recall |

        ### Kapan Menggunakan Metrik Apa?

        | Skenario | Metrik Utama | Alasan |
        |----------|-------------|--------|
        | Diagnosis penyakit | **Recall** | Jangan sampai ada pasien sakit yang terlewat |
        | Filter spam | **Precision** | Jangan sampai email penting masuk spam |
        | Balanced | **F1-Score** | Keseimbangan precision & recall |
        | Data seimbang | **Accuracy** | Proporsi kelas merata |
        """
    )


def _slide_confusion_matrix():
    st.markdown("## Confusion Matrix")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            <strong>Confusion Matrix</strong> adalah tabel yang menunjukkan performa model
            klasifikasi dengan membandingkan nilai <strong>aktual vs prediksi</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        |  | **Prediksi: Positif** | **Prediksi: Negatif** |
        |--|----------------------|----------------------|
        | **Aktual: Positif** | True Positive (TP) | False Negative (FN) |
        | **Aktual: Negatif** | False Positive (FP) | True Negative (TN) |

        ### Penjelasan
        - **TP**: Model prediksi Positif, dan memang Positif (benar)
        - **TN**: Model prediksi Negatif, dan memang Negatif (benar)
        - **FP**: Model prediksi Positif, padahal Negatif (Type I Error)
        - **FN**: Model prediksi Negatif, padahal Positif (Type II Error)
        """
    )

    st.markdown(
        """
        <div class="warning-box">
            <strong>Contoh Medis:</strong><br>
            FN (False Negative) = Pasien <strong>sakit</strong> tapi didiagnosis <strong>sehat</strong> — <strong>Berbahaya!</strong><br>
            FP (False Positive) = Pasien <strong>sehat</strong> tapi didiagnosis <strong>sakit</strong> — Perlu tes lanjutan
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_workflow():
    st.markdown("## Workflow Data Mining & Machine Learning")
    st.markdown("---")

    st.markdown(
        """
        ```
        +-------------------+
        | 1. PENGUMPULAN    |    Kumpulkan data dari sumber yang relevan
        |    DATA           |
        +---------+---------+
                  |
        +---------v---------+
        | 2. EKSPLORASI     |    EDA: Pahami distribusi, korelasi, outlier
        |    DATA (EDA)     |
        +---------+---------+
                  |
        +---------v---------+
        | 3. PREPROCESSING  |    Cleaning, encoding, scaling, split data
        |                   |
        +---------+---------+
                  |
        +---------v---------+
        | 4. PEMILIHAN &    |    Pilih algoritma, latih model
        |    TRAINING MODEL |
        +---------+---------+
                  |
        +---------v---------+
        | 5. EVALUASI       |    Accuracy, Precision, Recall, F1, CM
        |    MODEL          |
        +---------+---------+
                  |
        +---------v---------+
        | 6. DEPLOYMENT     |    Deploy model ke production
        |    & PREDIKSI     |
        +-------------------+
        ```

        > **Hari ini kita akan mempraktikkan semua langkah ini secara interaktif!**
        """
    )


def _slide_comparison():
    st.markdown("## Decision Tree vs Naive Bayes")
    st.markdown("---")

    st.markdown(
        """
        | Aspek | Decision Tree | Naive Bayes |
        |-------|--------------|-------------|
        | **Pendekatan** | Rule-based (if-then) | Probabilistik |
        | **Interpretasi** | Sangat mudah (visual tree) | Cukup mudah (probabilitas) |
        | **Kecepatan Training** | Sedang | Sangat Cepat |
        | **Kecepatan Prediksi** | Cepat | Sangat Cepat |
        | **Data Kecil** | Kurang baik | Baik |
        | **Data Besar** | Baik | Baik |
        | **Overfitting** | Rentan (tanpa pruning) | Tahan |
        | **Asumsi** | Tidak ada | Independensi fitur |
        | **Missing Values** | Bisa ditangani | Perlu handling |
        | **Fitur Interaksi** | Ya (natural) | Tidak |
        | **Normalisasi** | Tidak perlu | Tergantung tipe |
        | **Best Use Case** | Data terstruktur, interpretasi penting | Text, high-dim, baseline |
        """
    )


def _slide_real_world():
    st.markdown("## Aplikasi Dunia Nyata")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Decision Tree
            - **Diagnosis medis** — Sistem pendukung keputusan klinis
            - **Credit scoring** — Kelayakan kredit
            - **Quality control** — Deteksi produk cacat
            - **Customer churn** — Prediksi pelanggan keluar
            - **Pertanian** — Klasifikasi jenis tanah
            """
        )

    with col2:
        st.markdown(
            """
            ### Naive Bayes
            - **Email filtering** — Deteksi spam
            - **Sentiment analysis** — Analisis opini
            - **News classification** — Kategorisasi berita
            - **Medical diagnosis** — Screening awal
            - **Recommendation** — Sistem rekomendasi
            """
        )

    st.markdown(
        """
        <div class="success-box">
            <strong>Fun Fact:</strong> Google menggunakan variasi Naive Bayes untuk
            spam filtering di Gmail, dan Decision Tree (via Random Forest/XGBoost) banyak
            digunakan di kompetisi Kaggle!
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_demo_intro():
    st.markdown("## Saatnya Demo!")
    st.markdown("---")

    st.markdown(
        """
        <div style="text-align:center; padding:30px;">
            <h3 style="color:#3a86ff;">Mari kita praktikkan teori yang sudah dipelajari!</h3>
            <p style="font-size:1.1rem; color:#555;">
                Gunakan <strong>menu navigasi di sidebar</strong> untuk mengakses setiap tahap:
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### Eksplorasi Data
            Pahami data sebelum modeling

            ### Preprocessing
            Siapkan data untuk training
            """
        )

    with col2:
        st.markdown(
            """
            ### Pemodelan
            Training Decision Tree & Naive Bayes

            ### Perbandingan
            Bandingkan performa kedua model
            """
        )

    with col3:
        st.markdown(
            """
            ### Prediksi
            Coba prediksi data baru secara live!

            ### Soal Praktikum
            Uji pemahaman Anda!
            """
        )


def _slide_thank_you():
    st.markdown(
        """
        <div style="text-align:center; padding:50px 20px;">
            <h1 style="font-size:3rem; color:#1a1a2e;">Terima Kasih</h1>
            <h3 style="color:#555; font-weight:400; margin-top:1rem;">
                Semoga materi ini bermanfaat!
            </h3>
            <hr style="width:200px; margin:30px auto; border-color:#3a86ff;">
            <p style="font-size:1.1rem; color:#555;">
                <strong>Pemateri:</strong> Rahmad Ade Akbar, S.Pd., M.Pd<br>
                Mata Kuliah Data Mining
            </p>
            <br>
            <p style="font-size:1rem; color:#888;">
                <strong>Sesi Tanya Jawab</strong><br>
                Silakan ajukan pertanyaan
            </p>
            <br>
            <p style="font-size:0.9rem; color:#aaa;">
                GitHub: <a href="https://github.com/rahmadadeakbar" target="_blank">github.com/rahmadadeakbar</a><br>
                Source code tersedia di GitHub
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
