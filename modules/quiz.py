"""
Soal Praktikum — Perkuliahan Data Mining
Pemateri: Rahmad Ade Akbar, S.Pd., M.Pd
"""

import streamlit as st


def render_quiz():
    """Render interactive practicum quiz."""

    st.markdown("## Soal Praktikum Data Mining")
    st.markdown(
        """
        <div class="info-box">
            Jawablah soal-soal berikut untuk menguji pemahaman Anda tentang materi
            <strong>Decision Tree</strong> dan <strong>Naive Bayes</strong>.
            Klik tombol <strong>"Cek Jawaban"</strong> setelah menjawab semua soal.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    score = 0
    total = 10

    # ─── Soal 1 ────────────────────────────────────────────────
    st.markdown("### Soal 1 — Konsep Dasar Data Mining")
    q1 = st.radio(
        "Manakah yang paling tepat menggambarkan hubungan Data Mining dan Machine Learning?",
        [
            "a. Data Mining dan Machine Learning adalah hal yang sama",
            "b. Machine Learning adalah salah satu teknik yang digunakan dalam Data Mining",
            "c. Data Mining adalah bagian dari Machine Learning",
            "d. Keduanya tidak berhubungan sama sekali",
        ],
        index=None,
        key="q1",
    )

    # ─── Soal 2 ────────────────────────────────────────────────
    st.markdown("### Soal 2 — Tipe Pembelajaran")
    q2 = st.radio(
        "Decision Tree dan Naive Bayes termasuk dalam kategori pembelajaran apa?",
        [
            "a. Unsupervised Learning",
            "b. Reinforcement Learning",
            "c. Supervised Learning",
            "d. Semi-supervised Learning",
        ],
        index=None,
        key="q2",
    )

    # ─── Soal 3 ────────────────────────────────────────────────
    st.markdown("### Soal 3 — Decision Tree")
    q3 = st.radio(
        "Pada Decision Tree, kriteria splitting yang digunakan oleh Scikit-learn secara default adalah:",
        [
            "a. Information Gain",
            "b. Gain Ratio",
            "c. Gini Index",
            "d. Chi-Square",
        ],
        index=None,
        key="q3",
    )

    # ─── Soal 4 ────────────────────────────────────────────────
    st.markdown("### Soal 4 — Entropy")
    q4 = st.radio(
        "Jika nilai Entropy suatu node adalah 0, apa artinya?",
        [
            "a. Data pada node tersebut sangat campuran (mixed)",
            "b. Data pada node tersebut murni (hanya satu kelas)",
            "c. Model mengalami overfitting",
            "d. Node tersebut adalah root node",
        ],
        index=None,
        key="q4",
    )

    # ─── Soal 5 ────────────────────────────────────────────────
    st.markdown("### Soal 5 — Naive Bayes")
    q5 = st.radio(
        'Apa yang dimaksud dengan asumsi "naive" pada Naive Bayes?',
        [
            "a. Algoritma ini sangat sederhana dan tidak akurat",
            "b. Setiap fitur diasumsikan independen satu sama lain",
            "c. Model tidak menggunakan probabilitas",
            "d. Hanya bisa digunakan untuk data kecil",
        ],
        index=None,
        key="q5",
    )

    # ─── Soal 6 ────────────────────────────────────────────────
    st.markdown("### Soal 6 — Tipe Naive Bayes")
    q6 = st.radio(
        "Tipe Naive Bayes yang paling cocok untuk data numerik kontinu adalah:",
        [
            "a. Multinomial Naive Bayes",
            "b. Bernoulli Naive Bayes",
            "c. Gaussian Naive Bayes",
            "d. Complement Naive Bayes",
        ],
        index=None,
        key="q6",
    )

    # ─── Soal 7 ────────────────────────────────────────────────
    st.markdown("### Soal 7 — Evaluasi Model")
    q7 = st.radio(
        "Dalam kasus diagnosis penyakit, metrik evaluasi mana yang paling penting agar tidak ada pasien sakit yang terlewat?",
        [
            "a. Accuracy",
            "b. Precision",
            "c. Recall",
            "d. F1-Score",
        ],
        index=None,
        key="q7",
    )

    # ─── Soal 8 ────────────────────────────────────────────────
    st.markdown("### Soal 8 — Confusion Matrix")
    q8 = st.radio(
        "Pada Confusion Matrix, False Negative (FN) berarti:",
        [
            "a. Model memprediksi Positif, dan benar Positif",
            "b. Model memprediksi Negatif, padahal sebenarnya Positif",
            "c. Model memprediksi Positif, padahal sebenarnya Negatif",
            "d. Model memprediksi Negatif, dan benar Negatif",
        ],
        index=None,
        key="q8",
    )

    # ─── Soal 9 ────────────────────────────────────────────────
    st.markdown("### Soal 9 — Overfitting")
    q9 = st.radio(
        "Manakah solusi yang tepat untuk mengatasi overfitting pada Decision Tree?",
        [
            "a. Menambah lebih banyak fitur",
            "b. Menggunakan pruning (max_depth, min_samples_leaf)",
            "c. Menghilangkan semua fitur kecuali satu",
            "d. Tidak menggunakan split data (train/test)",
        ],
        index=None,
        key="q9",
    )

    # ─── Soal 10 ───────────────────────────────────────────────
    st.markdown("### Soal 10 — Perbandingan Model")
    q10 = st.radio(
        "Pernyataan mana yang BENAR tentang perbandingan Decision Tree dan Naive Bayes?",
        [
            "a. Decision Tree selalu lebih akurat daripada Naive Bayes",
            "b. Naive Bayes lebih rentan terhadap overfitting dibanding Decision Tree",
            "c. Decision Tree menggunakan pendekatan rule-based, Naive Bayes menggunakan pendekatan probabilistik",
            "d. Naive Bayes membutuhkan waktu training lebih lama daripada Decision Tree",
        ],
        index=None,
        key="q10",
    )

    # ─── Kunci Jawaban & Pembahasan ────────────────────────────
    st.markdown("---")

    answers = {
        "q1": "b. Machine Learning adalah salah satu teknik yang digunakan dalam Data Mining",
        "q2": "c. Supervised Learning",
        "q3": "c. Gini Index",
        "q4": "b. Data pada node tersebut murni (hanya satu kelas)",
        "q5": "b. Setiap fitur diasumsikan independen satu sama lain",
        "q6": "c. Gaussian Naive Bayes",
        "q7": "c. Recall",
        "q8": "b. Model memprediksi Negatif, padahal sebenarnya Positif",
        "q9": "b. Menggunakan pruning (max_depth, min_samples_leaf)",
        "q10": "c. Decision Tree menggunakan pendekatan rule-based, Naive Bayes menggunakan pendekatan probabilistik",
    }

    explanations = {
        "q1": "Machine Learning adalah teknik/alat utama yang digunakan dalam proses Data Mining untuk menemukan pola dari data.",
        "q2": "Keduanya termasuk Supervised Learning karena membutuhkan data berlabel (input dan output) untuk melatih model.",
        "q3": "Scikit-learn menggunakan Gini Index sebagai default criterion pada DecisionTreeClassifier, meskipun juga mendukung Entropy.",
        "q4": "Entropy = 0 berarti semua data pada node tersebut berada dalam satu kelas yang sama (murni/pure).",
        "q5": "Asumsi 'naive' berarti setiap fitur dianggap independen satu sama lain, sehingga P(X|C) = P(x1|C) x P(x2|C) x ... x P(xn|C).",
        "q6": "Gaussian NB mengasumsikan fitur berdistribusi normal (Gaussian), sehingga cocok untuk data numerik kontinu.",
        "q7": "Recall mengukur berapa banyak kasus positif yang berhasil terdeteksi. Dalam diagnosis penyakit, kita ingin meminimalkan False Negative.",
        "q8": "False Negative = model memprediksi Negatif padahal kenyataannya Positif. Contoh: pasien sakit didiagnosis sehat.",
        "q9": "Pruning membatasi pertumbuhan tree sehingga model tidak terlalu 'menghafal' data training (overfitting).",
        "q10": "Decision Tree membuat keputusan berbasis aturan if-then (rule-based), sedangkan Naive Bayes menghitung probabilitas kelas berdasarkan Teorema Bayes.",
    }

    user_answers = {
        "q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5,
        "q6": q6, "q7": q7, "q8": q8, "q9": q9, "q10": q10,
    }

    if st.button("Cek Jawaban", type="primary", use_container_width=True):
        st.session_state.quiz_submitted = True

    if st.session_state.quiz_submitted:
        st.markdown("---")
        st.markdown("## Hasil & Pembahasan")

        for i in range(1, total + 1):
            key = f"q{i}"
            user_ans = user_answers[key]
            correct_ans = answers[key]

            if user_ans is None:
                st.markdown(
                    f"""
                    <div class="warning-box">
                        <strong>Soal {i}:</strong> Belum dijawab<br>
                        <strong>Jawaban benar:</strong> {correct_ans}<br>
                        <strong>Penjelasan:</strong> {explanations[key]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif user_ans == correct_ans:
                score += 1
                st.markdown(
                    f"""
                    <div class="success-box">
                        <strong>Soal {i}: BENAR</strong><br>
                        <strong>Penjelasan:</strong> {explanations[key]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background:#fdecea; border-radius:6px; padding:1.2rem; margin:0.5rem 0; border-left:3px solid #e53e3e;">
                        <strong>Soal {i}: SALAH</strong><br>
                        <strong>Jawaban Anda:</strong> {user_ans}<br>
                        <strong>Jawaban benar:</strong> {correct_ans}<br>
                        <strong>Penjelasan:</strong> {explanations[key]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ─── Skor Akhir ───────────────────────────────────────
        st.markdown("---")

        pct = (score / total) * 100
        if pct >= 80:
            grade_color = "#2e8b3a"
            grade_label = "Sangat Baik"
        elif pct >= 60:
            grade_color = "#3a86ff"
            grade_label = "Baik"
        elif pct >= 40:
            grade_color = "#d69e2e"
            grade_label = "Cukup"
        else:
            grade_color = "#e53e3e"
            grade_label = "Perlu Belajar Lagi"

        st.markdown(
            f"""
            <div style="text-align:center; padding:30px; background:#f7f8fa; border-radius:8px; border:2px solid {grade_color};">
                <h2 style="color:{grade_color}; margin:0;">Skor Anda: {score} / {total}</h2>
                <h3 style="color:{grade_color}; margin:0.5rem 0;">{pct:.0f}%</h3>
                <p style="font-size:1.2rem; color:#555;">{grade_label}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Ulangi Quiz"):
            st.session_state.quiz_submitted = False
            for i in range(1, total + 1):
                if f"q{i}" in st.session_state:
                    del st.session_state[f"q{i}"]
            st.rerun()
