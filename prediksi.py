import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns

st.set_page_config(page_title="Prediksi Peminat Informatika", layout="wide")

st.title("👨‍💻 Aplikasi Prediksi Peminat Fakultas Teknik Informatika")
st.markdown("Menggunakan algoritma **Decision Tree** untuk memprediksi apakah seorang siswa akan memilih jurusan Informatika. Aplikasi ini adalah model prediksi berbasis algoritma Decision Tree yang dirancang untuk memperkirakan apakah seorang siswa akan memilih jurusan Teknik Informatika. Prediksi ini dibuat berdasarkan data siswa seperti jurusan SMA, nilai akademis, minat coding, dan kemampuan logika.")
st.markdown("Aplikasi ini memanfaatkan model machine learning, yaitu Decision Tree, untuk menganalisis pola dari data historis siswa. Fitur (input) yang digunakan untuk prediksi meliputi:")
st.markdown("Jurusan SMA (IPA, IPS, Bahasa")
st.markdown("Nilai Akademis (Matematika, Fisika, Bahasa Inggris")
st.markdown("Minat Coding (Tinggi, Sedang, Rendah")
st.markdown("Kemampuan Logika (dalam bentuk skor")
st.markdown("Jenis Kelamin")
st.markdown("Berdasarkan kombinasi dari data-data tersebut, model akan menghasilkan output berupa prediksi Ya atau Tidak, yang menunjukkan kecenderungan seorang siswa untuk memilih jurusan Informatika. Aplikasi ini bisa menjadi alat bantu yang berguna bagi institusi pendidikan untuk menyaring calon mahasiswa potensial atau bagi siswa untuk mengevaluasi pilihan karir mereka.")
# --- 1. Load & Tampilkan Dataset ---
try:
    df = pd.read_csv("dataset_prediksi_mahasiswa.csv")
    st.subheader("📋 Dataset Awal")
    st.dataframe(df)
except FileNotFoundError:
    st.error("File 'dataset_prediksi_mahasiswa.csv' tidak ditemukan. Pastikan file tersebut berada di folder yang sama dengan aplikasi ini.")
    st.stop()

# --- 2. Preprocessing Data ---
# Drop kolom ID_Siswa karena tidak relevan untuk prediksi
df_processed = df.drop('ID_Siswa', axis=1)

# Pisahkan fitur (X) dan target (y)
target = 'Pilih_Informatika'
features = df_processed.drop(target, axis=1)
y = df_processed[target]

# One-Hot Encoding untuk semua fitur kategorikal
# Ini mengubah kolom seperti 'Jurusan_SMA' menjadi numerik
X_encoded = pd.get_dummies(features, columns=['Jurusan_SMA', 'Minat_Coding', 'Jenis_Kelamin'])

st.subheader("📊 Data Setelah Preprocessing")
st.caption("Kolom ID_Siswa dihapus dan fitur kategorikal diubah menjadi format numerik.")
st.dataframe(X_encoded.head())

# --- 3. Split Data & Train Model ---
# Menggunakan stratify=y agar proporsi 'Ya' dan 'Tidak' seimbang di data train dan test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# --- 4. Evaluasi Model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
class_labels = model.classes_ # Akan berisi ['Tidak', 'Ya']

st.subheader("📈 Evaluasi Kinerja Model")
st.write(f"**Akurasi Model:** {accuracy * 100:.2f}%")

col1, col2 = st.columns(2)

with col1:
    # Classification report
    st.text("Classification Report:")
    report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

with col2:
    # Confusion matrix
    st.write("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Greens", xticklabels=class_labels, yticklabels=class_labels, ax=ax_cm)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    st.pyplot(fig_cm)


# --- 5. Input Manual untuk Prediksi ---
st.subheader("📝 Coba Prediksi dengan Data Baru")
st.markdown("Masukkan data calon siswa untuk melihat prediksi:")

# Buat form untuk input yang lebih terstruktur
with st.form("prediction_form"):
    col_input1, col_input2 = st.columns(2)

    with col_input1:
        jurusan_sma = st.selectbox("Jurusan SMA", df['Jurusan_SMA'].unique())
        minat_coding = st.selectbox("Minat Coding", df['Minat_Coding'].unique())
        jenis_kelamin = st.selectbox("Jenis Kelamin", df['Jenis_Kelamin'].unique())
        
    with col_input2:
        nilai_mat = st.slider("Nilai Matematika", 0, 100, 85)
        nilai_fis = st.slider("Nilai Fisika", 0, 100, 80)
        kemampuan_logika = st.slider("Kemampuan Logika (1-10)", 1, 10, 8)
        # Nilai Bhs Inggris tidak ada di slider karena kurang relevan di tree, tapi kita tetap butuh datanya
        nilai_ing = st.slider("Nilai Bahasa Inggris", 0, 100, 85)


    # Tombol submit ada di dalam form
    submitted = st.form_submit_button("🔮 Prediksi Sekarang")

if submitted:
    # Buat DataFrame dari input user
    input_data = {
        'Jurusan_SMA': [jurusan_sma],
        'Nilai_Matematika': [nilai_mat],
        'Nilai_Fisika': [nilai_fis],
        'Nilai_Bhs_Inggris': [nilai_ing],
        'Minat_Coding': [minat_coding],
        'Kemampuan_Logika': [kemampuan_logika],
        'Jenis_Kelamin': [jenis_kelamin]
    }
    input_df = pd.DataFrame(input_data)

    # Lakukan One-Hot Encoding pada input user
    input_encoded = pd.get_dummies(input_df)

    # Rekonstruksi kolom agar sama persis dengan data training
    input_final = input_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Lakukan prediksi
    hasil_prediksi = model.predict(input_final)[0]
    hasil_proba = model.predict_proba(input_final)[0]

    st.subheader("✔️ Hasil Prediksi")
    if hasil_prediksi == "Ya":
        st.success(f"Berdasarkan data yang dimasukkan, siswa ini **BERPOTENSI MEMILIH** Teknik Informatika.")
    else:
        st.error(f"Berdasarkan data yang dimasukkan, siswa ini **KURANG BERPOTENSI MEMILIH** Teknik Informatika.")

    # Tampilkan probabilitas
    st.write("Probabilitas:")
    proba_df = pd.DataFrame([hasil_proba], columns=model.classes_, index=["Peluang"])
    st.dataframe(proba_df)


# --- 6. Visualisasi Decision Tree ---
st.subheader("🌳 Visualisasi Pohon Keputusan (Decision Tree)")
st.caption("Diagram ini menunjukkan alur logika yang digunakan model untuk mengambil keputusan.")
with st.expander("Klik untuk melihat detail pohon keputusan"):
    fig_tree, ax_tree = plt.subplots(figsize=(25, 15))
    tree.plot_tree(model,
                   feature_names=X_encoded.columns,
                   class_names=model.classes_,
                   filled=True,
                   rounded=True,
                   fontsize=10)
    st.pyplot(fig_tree)