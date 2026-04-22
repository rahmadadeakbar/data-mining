"""
Latihan Praktikum — Perkuliahan Data Mining
Pemateri: Rahmad Ade Akbar, S.Pd., M.Pd

Modul latihan praktikum step-by-step menggunakan Decision Tree & Naive Bayes
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff


def render_practicum(df, feature_names, target_name, target_labels, dataset_choice):
    """Render interactive practicum exercise with step-by-step guide."""

    st.markdown("## Latihan Praktikum Data Mining")
    st.markdown(
        """
        <div class="info-box">
            Ikuti langkah-langkah praktikum berikut secara berurutan.
            Setiap tahap memiliki <strong>penjelasan teori singkat</strong>,
            <strong>kode Python</strong> yang bisa dipelajari, dan
            <strong>output interaktif</strong> untuk diamati.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"**Dataset aktif:** `{dataset_choice}` — {len(df)} baris, {len(feature_names)} fitur")
    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # TAHAP 1 — Memahami Dataset
    # ═══════════════════════════════════════════════════════════
    st.markdown("### Tahap 1 — Memahami Dataset")
    st.markdown(
        """
        <div class="info-box">
            <strong>Tujuan:</strong> Mengenal struktur data sebelum melakukan pemodelan.
            Pahami jumlah baris, kolom, tipe data, dan distribusi kelas target.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Lihat Kode Python", expanded=False):
        st.code(
            """
import pandas as pd
from sklearn.datasets import load_iris  # atau dataset lainnya

# Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Lihat struktur data
print(df.shape)          # Jumlah baris & kolom
print(df.info())         # Tipe data
print(df.describe())     # Statistik deskriptif
print(df['target'].value_counts())  # Distribusi kelas
            """,
            language="python",
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"**Dimensi data:** `{df.shape[0]}` baris x `{df.shape[1]}` kolom")
        st.markdown(f"**Fitur:** `{len(feature_names)}` fitur")
        st.markdown(f"**Target:** `{target_name}`")
        if target_labels:
            st.markdown(
                f"**Kelas:** {', '.join(str(l) for l in target_labels)}")
    with col2:
        st.markdown("**Distribusi Kelas Target:**")
        class_dist = df[target_name].value_counts()
        st.dataframe(class_dist.reset_index().rename(columns={
                     "index": "Kelas", target_name: "Kelas", "count": "Jumlah"}), hide_index=True)

    st.markdown("**5 Baris Pertama:**")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown(
        """
        <div class="success-box">
            <strong>Tugas 1:</strong> Perhatikan distribusi kelas target.
            Apakah dataset ini seimbang (balanced) atau tidak seimbang (imbalanced)?
            Catat jawaban Anda.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # TAHAP 2 — Statistik Deskriptif & Visualisasi
    # ═══════════════════════════════════════════════════════════
    st.markdown("### Tahap 2 — Statistik Deskriptif & Visualisasi")
    st.markdown(
        """
        <div class="info-box">
            <strong>Tujuan:</strong> Memahami distribusi setiap fitur dan hubungan antar variabel
            melalui statistik deskriptif dan visualisasi sederhana.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Lihat Kode Python", expanded=False):
        st.code(
            """
# Statistik deskriptif
print(df.describe())

# Cek missing values
print(df.isnull().sum())

# Visualisasi distribusi fitur
import matplotlib.pyplot as plt
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Korelasi antar fitur
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
            """,
            language="python",
        )

    st.markdown("**Statistik Deskriptif:**")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("**Missing Values:**")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.markdown(
            '<div class="success-box">Tidak ada missing values pada dataset ini.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(missing[missing > 0])

    # Visualisasi distribusi fitur pertama
    if len(feature_names) >= 2:
        fig = px.histogram(
            df,
            x=feature_names[0],
            color=target_name,
            title=f"Distribusi {feature_names[0]} berdasarkan Kelas Target",
            barmode="overlay",
            opacity=0.7,
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="success-box">
            <strong>Tugas 2:</strong> Dari statistik deskriptif, identifikasi fitur mana yang
            memiliki range nilai paling besar. Apakah perlu dilakukan scaling? Mengapa?
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # TAHAP 3 — Preprocessing Data
    # ═══════════════════════════════════════════════════════════
    st.markdown("### Tahap 3 — Preprocessing Data")
    st.markdown(
        """
        <div class="info-box">
            <strong>Tujuan:</strong> Menyiapkan data untuk proses training model.
            Langkah utama: memisahkan fitur & target, membagi data train/test, dan melakukan scaling.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Lihat Kode Python", expanded=False):
        st.code(
            """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Pisahkan fitur (X) dan target (y)
X = df[feature_names]
y = df[target_name]

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training: {X_train.shape[0]} data")
print(f"Testing:  {X_test.shape[0]} data")

# Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
            """,
            language="python",
        )

    col_ts, col_rs = st.columns(2)
    with col_ts:
        test_size = st.slider("Test Size (%)", 10, 40,
                              20, 5, key="prac_test_size")
    with col_rs:
        random_state = st.number_input(
            "Random State", 0, 100, 42, key="prac_rs")

    X = df[feature_names].values
    y = df[target_name].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data", len(df))
    with col2:
        st.metric("Data Training", len(X_train))
    with col3:
        st.metric("Data Testing", len(X_test))

    st.markdown(
        """
        <div class="success-box">
            <strong>Tugas 3:</strong> Coba ubah test size menjadi 30%.
            Apa yang berubah pada jumlah data training dan testing?
            Menurut Anda, rasio berapa yang ideal dan mengapa?
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # TAHAP 4 — Training Decision Tree
    # ═══════════════════════════════════════════════════════════
    st.markdown("### Tahap 4 — Training Model Decision Tree")
    st.markdown(
        """
        <div class="info-box">
            <strong>Tujuan:</strong> Melatih model Decision Tree dan melihat hasil prediksi
            serta evaluasinya.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Lihat Kode Python", expanded=False):
        st.code(
            """
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Buat dan latih model Decision Tree
dt_model = DecisionTreeClassifier(
    criterion='gini',      # atau 'entropy'
    max_depth=5,           # batasi kedalaman tree
    random_state=42
)
dt_model.fit(X_train, y_train)

# Prediksi
y_pred_dt = dt_model.predict(X_test)

# Evaluasi
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(classification_report(y_test, y_pred_dt))
            """,
            language="python",
        )

    col_cr, col_md = st.columns(2)
    with col_cr:
        dt_criterion = st.selectbox(
            "Criterion", ["gini", "entropy"], key="prac_dt_crit")
    with col_md:
        dt_max_depth = st.slider("Max Depth", 1, 20, 5, key="prac_dt_depth")

    dt_model = DecisionTreeClassifier(
        criterion=dt_criterion, max_depth=dt_max_depth, random_state=random_state
    )
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    dt_acc = accuracy_score(y_test, y_pred_dt)
    st.metric("Accuracy Decision Tree", f"{dt_acc:.4f}")

    st.markdown("**Classification Report:**")
    dt_report = classification_report(y_test, y_pred_dt, output_dict=True)
    st.dataframe(pd.DataFrame(dt_report).T.round(4), use_container_width=True)

    # Confusion Matrix
    dt_cm = confusion_matrix(y_test, y_pred_dt)
    labels = [str(l) for l in sorted(np.unique(y_test))]
    fig_dt_cm = ff.create_annotated_heatmap(
        dt_cm, x=labels, y=labels,
        colorscale="Blues", showscale=True,
    )
    fig_dt_cm.update_layout(
        title="Confusion Matrix — Decision Tree",
        xaxis_title="Prediksi", yaxis_title="Aktual", height=350,
    )
    st.plotly_chart(fig_dt_cm, use_container_width=True)

    # Feature importance
    fi = pd.DataFrame({
        "Fitur": feature_names,
        "Importance": dt_model.feature_importances_,
    }).sort_values("Importance", ascending=True)
    fig_fi = px.bar(fi, x="Importance", y="Fitur", orientation="h",
                    title="Feature Importance — Decision Tree")
    fig_fi.update_layout(height=max(300, len(feature_names) * 25))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(
        """
        <div class="success-box">
            <strong>Tugas 4:</strong> Coba ubah criterion dari 'gini' ke 'entropy'.
            Apakah ada perbedaan accuracy? Coba juga ubah max_depth menjadi 3 dan 10.
            Apa yang terjadi? Jelaskan hubungannya dengan overfitting.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # TAHAP 5 — Training Naive Bayes
    # ═══════════════════════════════════════════════════════════
    st.markdown("### Tahap 5 — Training Model Naive Bayes")
    st.markdown(
        """
        <div class="info-box">
            <strong>Tujuan:</strong> Melatih model Gaussian Naive Bayes pada data yang
            sudah di-scaling, lalu mengevaluasi performanya.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Lihat Kode Python", expanded=False):
        st.code(
            """
from sklearn.naive_bayes import GaussianNB

# Buat dan latih model Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)  # Gunakan data yang sudah di-scaling

# Prediksi
y_pred_nb = nb_model.predict(X_test_scaled)

# Evaluasi
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(classification_report(y_test, y_pred_nb))
            """,
            language="python",
        )

    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    y_pred_nb = nb_model.predict(X_test_scaled)

    nb_acc = accuracy_score(y_test, y_pred_nb)
    st.metric("Accuracy Naive Bayes", f"{nb_acc:.4f}")

    st.markdown("**Classification Report:**")
    nb_report = classification_report(y_test, y_pred_nb, output_dict=True)
    st.dataframe(pd.DataFrame(nb_report).T.round(4), use_container_width=True)

    # Confusion Matrix
    nb_cm = confusion_matrix(y_test, y_pred_nb)
    fig_nb_cm = ff.create_annotated_heatmap(
        nb_cm, x=labels, y=labels,
        colorscale="Oranges", showscale=True,
    )
    fig_nb_cm.update_layout(
        title="Confusion Matrix — Naive Bayes",
        xaxis_title="Prediksi", yaxis_title="Aktual", height=350,
    )
    st.plotly_chart(fig_nb_cm, use_container_width=True)

    st.markdown(
        """
        <div class="success-box">
            <strong>Tugas 5:</strong> Bandingkan Confusion Matrix Decision Tree vs Naive Bayes.
            Model mana yang memiliki lebih sedikit kesalahan? Pada kelas mana
            masing-masing model lebih unggul?
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # TAHAP 6 — Perbandingan Kedua Model
    # ═══════════════════════════════════════════════════════════
    st.markdown("### Tahap 6 — Perbandingan Kedua Model")
    st.markdown(
        """
        <div class="info-box">
            <strong>Tujuan:</strong> Membandingkan performa Decision Tree dan Naive Bayes
            secara langsung menggunakan berbagai metrik.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Lihat Kode Python", expanded=False):
        st.code(
            """
# Bandingkan accuracy
results = {
    'Model': ['Decision Tree', 'Naive Bayes'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_nb),
    ],
}
comparison = pd.DataFrame(results)
print(comparison)

# Visualisasi perbandingan
import matplotlib.pyplot as plt
comparison.plot(x='Model', y='Accuracy', kind='bar')
plt.title('Perbandingan Accuracy')
plt.ylim(0, 1)
plt.show()
            """,
            language="python",
        )

    # Metrics comparison
    dt_full = classification_report(y_test, y_pred_dt, output_dict=True)
    nb_full = classification_report(y_test, y_pred_nb, output_dict=True)

    comparison_df = pd.DataFrame({
        "Metrik": ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-Score (weighted)"],
        "Decision Tree": [
            dt_acc,
            dt_full["weighted avg"]["precision"],
            dt_full["weighted avg"]["recall"],
            dt_full["weighted avg"]["f1-score"],
        ],
        "Naive Bayes": [
            nb_acc,
            nb_full["weighted avg"]["precision"],
            nb_full["weighted avg"]["recall"],
            nb_full["weighted avg"]["f1-score"],
        ],
    })
    comparison_df["Selisih"] = (
        comparison_df["Decision Tree"] - comparison_df["Naive Bayes"]).round(4)
    st.dataframe(comparison_df.round(
        4), use_container_width=True, hide_index=True)

    # Bar chart comparison
    fig_comp = px.bar(
        comparison_df.melt(id_vars="Metrik", value_vars=["Decision Tree", "Naive Bayes"],
                           var_name="Model", value_name="Skor"),
        x="Metrik", y="Skor", color="Model", barmode="group",
        title="Perbandingan Metrik Evaluasi",
    )
    fig_comp.update_layout(height=400, yaxis_range=[0, 1])
    st.plotly_chart(fig_comp, use_container_width=True)

    # Determine winner
    if dt_acc > nb_acc:
        winner = "Decision Tree"
        diff = dt_acc - nb_acc
    elif nb_acc > dt_acc:
        winner = "Naive Bayes"
        diff = nb_acc - dt_acc
    else:
        winner = "Keduanya Sama"
        diff = 0

    if winner != "Keduanya Sama":
        st.markdown(
            f"""
            <div class="success-box">
                <strong>Hasil:</strong> Model <strong>{winner}</strong> lebih unggul
                dengan selisih accuracy <strong>{diff:.4f}</strong> pada dataset {dataset_choice}.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="success-box">
                <strong>Hasil:</strong> Kedua model memiliki accuracy yang <strong>sama</strong>
                pada dataset ini.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="success-box">
            <strong>Tugas 6:</strong> Berdasarkan semua metrik evaluasi di atas, model mana
            yang Anda rekomendasikan untuk dataset ini? Jelaskan alasan Anda berdasarkan:
            <ol>
                <li>Accuracy</li>
                <li>Precision vs Recall (mana yang lebih penting untuk domain ini?)</li>
                <li>Kecepatan dan interpretability</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # TAHAP 7 — Kesimpulan & Refleksi
    # ═══════════════════════════════════════════════════════════
    st.markdown("### Tahap 7 — Kesimpulan & Refleksi")
    st.markdown(
        """
        <div class="info-box">
            <strong>Tujuan:</strong> Merangkum hasil praktikum dan merefleksikan
            apa yang telah dipelajari.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Ringkasan Tahapan Praktikum

        | Tahap | Kegiatan | Hasil |
        |-------|----------|-------|
        | 1 | Memahami Dataset | Mengetahui struktur, dimensi, dan distribusi kelas |
        | 2 | Statistik Deskriptif | Memahami distribusi fitur dan korelasi |
        | 3 | Preprocessing | Split data dan scaling |
        | 4 | Decision Tree | Training, evaluasi, dan feature importance |
        | 5 | Naive Bayes | Training dan evaluasi dengan data ter-scaling |
        | 6 | Perbandingan | Membandingkan performa kedua model |
        | 7 | Kesimpulan | Refleksi dan rekomendasi |

        ### Pertanyaan Refleksi
        1. Apa perbedaan mendasar antara cara Decision Tree dan Naive Bayes membuat keputusan?
        2. Mengapa scaling penting untuk Naive Bayes tapi tidak untuk Decision Tree?
        3. Dalam kondisi apa Anda akan memilih Decision Tree? Dalam kondisi apa memilih Naive Bayes?
        4. Apa yang bisa dilakukan untuk meningkatkan performa model?
        """
    )

    st.markdown(
        """
        <div class="warning-box">
            <strong>Tugas Akhir:</strong> Buatlah laporan singkat (1 halaman) yang berisi:
            <ol>
                <li>Dataset yang digunakan dan karakteristiknya</li>
                <li>Hasil evaluasi Decision Tree dan Naive Bayes</li>
                <li>Perbandingan dan analisis: model mana yang lebih baik dan mengapa</li>
                <li>Kesimpulan dan saran untuk peningkatan performa</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
