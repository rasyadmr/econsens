import streamlit as st
import pandas as pd
import numpy as np
import torch
import html
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import string
import time
from model import IndoBERTCustom
from datasets import load_dataset

st.set_page_config(
    page_title="EconSens - Klasifikasi Sentimen Ekonomi Indonesia",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'page' not in st.session_state:
    st.session_state.page = 'utama'
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'preprocessing_steps' not in st.session_state:
    st.session_state.preprocessing_steps = []

def halaman_utama():
    st.title("EconSens")
    st.markdown("### Aplikasi Klasifikasi Sentimen Perekonomian Indonesia")

    col1, col2 = st.columns(2)

    st.markdown("""
            #### Penjelasan Aplikasi EconSens

            EconSens merupakan aplikasi klasifikasi sentimen yang dirancang khusus untuk
            menganalisis opini publik tentang perekonomian Indonesia dari media sosial
            menggunakan model IndoBERT yang telah dilatih. Aplikasi ini dapat memudahkan Anda
            dalam melakukan klasifikasi sentimen pada topik ekonomi Indonesia secara cepat
            dan akurat tanpa perlu mengeluarkan biaya apapun.
            """)

    st.markdown("""
            #### Penjelasan Eksperimen Model

            Model ini dilatih menggunakan dataset tweet ekonomi Indonesia dengan
            membandingkan 4 model: Logistic Regression, SVM, LSTM, dan IndoBERT,
            di mana IndoBERT menunjukkan performa terbaik. Model IndoBERT telah dilatih menggunakan
            lebih dari 5000 data dari media sosial X dengan akurasi yang didapatkan
            sebesar sekitar `94%`.
            """)
    # with col1:
    #     with st.container():
    #         st.markdown("""
    #         #### Penjelasan Aplikasi EconSens

    #         EcoSens merupakan aplikasi klasifikasi sentimen yang dirancang khusus untuk
    #         menganalisis opini publik tentang perekonomian Indonesia dari media sosial
    #         menggunakan model IndoBERT yang telah dilatih.
    #         """)

    # with col2:
    #     with st.container():
    #         st.markdown("""
    #         #### Penjelasan Eksperimen Model

    #         Model ini dilatih menggunakan dataset tweet ekonomi Indonesia dengan
    #         membandingkan 4 model: Logistic Regression, SVM, LSTM, dan IndoBERT,
    #         di mana IndoBERT menunjukkan performa terbaik.
    #         """)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Panduan Penggunaan", width="stretch", type="primary"):
            st.session_state.page = 'panduan'
            st.rerun()
    with col2:
        if st.button("Klasifikasi Sentimen", width="stretch", type="primary"):
            st.session_state.page = 'prediksi'
            st.rerun()

@st.cache_resource
def load_model():
    try:
        MODEL_PATH = "./best_indobert_model.pth"
        TOKENIZER_NAME = "indobenchmark/indobert-base-p1"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            model_data = torch.load(MODEL_PATH, map_location=device)
        except Exception as e:
            st.error(f"Tidak dapat membaca file model: {str(e)}")
            return None, None, None

        if 'model_config' not in model_data:
            st.error("File model tidak memiliki konfigurasi yang diperlukan")
            return None, None, None

        model_config = model_data['model_config']
        NUM_LABELS = model_config.get('num_labels', 2)
        DROPOUT_RATE = model_config.get('dropout_rate', 0.3)
        HIDDEN_DIM = model_config.get('hidden_dim', 256)

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        config = AutoConfig.from_pretrained(
            TOKENIZER_NAME,
            num_labels=NUM_LABELS,
            return_dict=True
        )

        model = IndoBERTCustom.from_pretrained(
            TOKENIZER_NAME,
            config=config,
            dropout_rate=DROPOUT_RATE,
            hidden_dim=HIDDEN_DIM
        )

        # Load state dict
        if 'model_state_dict' in model_data:
            model.load_state_dict(model_data['model_state_dict'], strict=True)
        else:
            st.error("File model tidak memiliki model_state_dict")
            return None, None, None

        model.eval()
        model = model.to(device)

        return tokenizer, model, device

    except Exception as e:
        st.error(f"Error memuat model: {str(e)}")
        return None, None, None

@st.cache_resource
def load_normalization_dataset():
    try:
        slang_dict = {}
        try:
            slang_ds = load_dataset("zeroix07/indo-slang-words", split="train")
            for item in slang_ds['text']:
                parts = item.split(':', 1)
                if len(parts) == 2:
                    slang = parts[0].strip()
                    formal = parts[1].strip()
                    if slang and formal:
                        slang_dict[slang] = formal
        except Exception as e:
            st.warning(f"Tidak dapat memuat dataset slang, menggunakan manual dict saja: {str(e)}")

        manual_normalization_dict = {
            'yg': 'yang', 'jg': 'juga', 'ga': 'tidak', 'gak': 'tidak', 'enggak': 'tidak',
            'nya': '', 'polri': 'polisi', 'kpd': 'kepada', 'dgn': 'dengan',
            'utk': 'untuk', 'tks': 'terima kasih', 'krn': 'karena', 'dlm': 'dalam',
            'lalin': 'lalu lintas', 'smg': 'semoga', 'sll': 'selalu', 'thx': 'terima kasih',
            'gue': 'saya', 'jd': 'jadi', 'lgsg': 'langsung', 'tp': 'tapi', 'tgl': 'tanggal',
            'hr': 'hari', 'blm': 'belum', 'sdh': 'sudah', 'mk': 'maka', 'bg': 'bagi'
        }

        normalization_dict = {**slang_dict, **manual_normalization_dict}
        return normalization_dict
    except Exception as e:
        st.error(f"Error memuat normalization dataset: {str(e)}")
        return {}

def normalize_text(text, normalization_dict):
    words = text.split()
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

def preprocess_text(text, normalization_dict):
    steps = []

    text = str(text)
    steps.append(("Teks asli", text))

    text = text.lower()
    steps.append(("Case folding", text))

    text = html.unescape(text)
    steps.append(("HTML Unescape", text))

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    steps.append(("Menghilangkan URL", text))

    text = re.sub(r'@\w+|#\w+', '', text)
    steps.append(("Menghilangkan mentions/hashtags", text))

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    steps.append(("Menghilangkan simbol & angka", text))

    text = ' '.join(text.split())
    steps.append(("Menghilangkan extra space", text))

    if normalization_dict:
        text = normalize_text(text, normalization_dict)
        steps.append(("Normalisasi text", text))

    return text, steps

def predict_sentiment_torch(text, tokenizer, model, device):
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).cpu().numpy()[0]
            confidence_scores = probs.cpu().numpy()[0]

        label_map = {0: "Negatif", 1: "Positif"}
        sentiment = label_map[predicted_class]
        confidence = float(confidence_scores[predicted_class])

        return {
            'label': sentiment,
            'confidence': confidence,
            'scores': {
                'Positif': float(confidence_scores[1]),
                'Negatif': float(confidence_scores[0]),
            }
        }

    except Exception as e:
        return {'label': 'Error', 'confidence': 0.0, 'scores': {'Positif': 0, 'Negatif': 0}, 'error': str(e)}

def validasi_teks(text):
    if not text or len(text.strip()) == 0:
        return False, "Teks tidak boleh kosong"
    if len(text) > 5000:
        return False, "Teks terlalu panjang (maksimal 5000 karakter)"
    if len(text.split()) < 3:
        return False, "Teks terlalu pendek (minimal 3 kata)"
    return True, "Valid"

def validasi_file(file):
    valid_extensions = ['txt', 'csv', 'xlsx']
    file_ext = file.name.split('.')[-1].lower()

    if file_ext not in valid_extensions:
        return False, f"Format file tidak didukung. Gunakan: {', '.join(valid_extensions)}"
    if file.size > 10 * 1024 * 1024:
        return False, "Ukuran file terlalu besar (maksimal 10MB)"
    return True, "Valid"

def create_visualization(predictions):
    if not predictions:
        return None, None

    valid_preds = [p for p in predictions if p.get('label') in ['Positif', 'Negatif']]

    if not valid_preds:
        return None, None

    sentiments = [p['label'] for p in valid_preds]
    sentiment_counts = pd.Series(sentiments).value_counts()

    colors = {'Positif': '#00CC88', 'Negatif': '#FF4B4B'}

    fig_pie = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=.3,
        marker=dict(colors=[colors.get(label, '#808080') for label in sentiment_counts.index])
    )])
    fig_pie.update_layout(
        title="Distribusi Sentimen",
        height=400
    )

    avg_confidence = {}
    for sentiment in ['Positif', 'Negatif']:
        scores = [p['confidence'] for p in valid_preds if p['label'] == sentiment]
        if scores:
            avg_confidence[sentiment] = np.mean(scores)

    if avg_confidence:
        fig_bar = go.Figure(data=[
            go.Bar(
                x=list(avg_confidence.keys()),
                y=list(avg_confidence.values()),
                marker_color=[colors.get(s, '#808080') for s in avg_confidence.keys()],
                text=[f'{v:.1%}' for v in avg_confidence.values()],
                textposition='auto'
            )
        ])
        fig_bar.update_layout(
            title="Rata-rata Confidence Score",
            yaxis_title="Confidence",
            yaxis_range=[0, 1],
            height=400
        )
        return fig_pie, fig_bar
    else:
        return fig_pie, None

with st.sidebar:
    st.title("EconSens", text_alignment="center")
    st.divider()

    if st.button("Halaman Utama", width="stretch", type= "primary" if st.session_state.page == "utama" else "secondary"):
        st.session_state.page = "utama"
        st.rerun()

    if st.button("Prediksi Sentimen", width="stretch", type= "primary" if st.session_state.page == "prediksi" else "secondary"):
        st.session_state.page = "prediksi"
        st.rerun()

    if st.button("Panduan Penggunaan", width="stretch", type= "primary" if st.session_state.page == "panduan" else "secondary"):
        st.session_state.page = 'panduan'
        st.rerun()

    # st.divider()
    # st.markdown("### Info Model")
    # st.info("Model: IndoBERT\nAkurasi: >85%")

def process_single_text(text, normalization_dict, tokenizer, model, device):
    processed_text, prep_steps = preprocess_text(text, normalization_dict)

    if not processed_text or not processed_text.strip():
        return None, prep_steps, "Teks tidak dapat diklasifikasi karena hasil preprocessing kosong"

    result = predict_sentiment_torch(processed_text, tokenizer, model, device)
    if result and not result.get('error'):
        return result, prep_steps, None

    return None, prep_steps, "Terjadi kesalahan saat prediksi"

def process_file_texts(texts, normalization_dict, tokenizer, model, device):
    results = []
    skipped_count = 0
    progress_bar = st.progress(0)

    for i, text in enumerate(texts):
        progress_bar.progress((i + 1) / len(texts))

        if not text.strip():
            continue

        processed_text, _ = preprocess_text(str(text), normalization_dict)

        if not processed_text or not processed_text.strip():
            skipped_count += 1
            results.append({
                'Text': text[:100] + "..." if len(text) > 100 else text,
                'Sentiment': 'Tidak dapat diklasifikasi',
                'Confidence': 'N/A'
            })
            continue

        result = predict_sentiment_torch(processed_text, tokenizer, model, device)

        if not result or result.get('error'):
            continue

        results.append({
            'Text': text[:100] + "..." if len(text) > 100 else text,
            'Sentiment': result['label'],
            'Confidence': f"{result['confidence']:.2%}"
        })

        st.session_state.predictions.append({
            'text': text[:100] + "...",
            'label': result['label'],
            'confidence': result['confidence'],
            'timestamp': datetime.now()
        })
    return results, skipped_count

def read_file_contents(uploaded_file):
    """Read and parse uploaded file contents"""
    try:
        if uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode('utf-8')
            texts = [line.strip() for line in content.split('\n') if line.strip()]
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            texts = df.iloc[:, 0].dropna().astype(str).tolist()
        else:
            df = pd.read_excel(uploaded_file)
            texts = df.iloc[:, 0].dropna().astype(str).tolist()
        return texts, None
    except Exception as e:
        return None, f"Error membaca file: {str(e)}"

def display_single_result(result):
    """Display single prediction result"""
    col1_res, col2_res = st.columns([2, 1])
    with col1_res:
        st.markdown("### Hasil Prediksi")
        if result['label'] == 'Positif':
            st.success(f"**Sentimen: {result['label']}** (Confidence: {result['confidence']:.2%})")
        elif result['label'] == 'Negatif':
            st.error(f"**Sentimen: {result['label']}** (Confidence: {result['confidence']:.2%})")

    with col2_res:
        st.markdown("### Confidence Scores")
        st.metric("Positif", f"{result['scores']['Positif']:.2%}")
        st.metric("Negatif", f"{result['scores']['Negatif']:.2%}")

def display_batch_results(results, skipped_count, total_count):
    """Display batch processing results"""
    st.success(f"Selesai! Berhasil memproses {total_count - skipped_count} data.")
    if skipped_count > 0:
        st.warning(f"⚠️ {skipped_count} data dilewati karena teks kosong setelah preprocessing.")

    st.markdown("### Hasil Analisis")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

def display_preprocessing_steps(preprocessing_steps):
    """Display preprocessing steps in expander"""
    with st.expander("Lihat Tahapan Preprocessing", expanded=False):
        for step, text in preprocessing_steps:
            st.text(f"{step}:")
            st.code(text if text else "(empty)", language=None)

def display_visualizations_and_download():
    if st.session_state.predictions:
        st.divider()
        st.markdown("### Visualisasi Prediksi")

        try:
            fig_pie, fig_bar = create_visualization(st.session_state.predictions)

            col1_vis, col2_vis = st.columns(2)
            with col1_vis:
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("Belum ada data visualisasi sentimen.")
            with col2_vis:
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("Belum ada data visualisasi confidence.")
        except Exception as e:
            st.info("Data belum cukup untuk visualisasi.")

        # Download button
        st.divider()
        df_download = pd.DataFrame(st.session_state.predictions)
        if not df_download.empty and 'timestamp' in df_download.columns:
            df_download['timestamp'] = df_download['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            csv = df_download.to_csv(index=False).encode('utf-8')

            _, col_download, _ = st.columns([1, 2, 1])
            with col_download:
                st.download_button(
                    label="Download Hasil",
                    data=csv,
                    file_name=f"hasil_prediksi_econsens_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv',
                    type="primary",
                    use_container_width=True
                )

def halaman_prediksi():
    st.title("Prediksi Sentimen")

    tokenizer, model, device = load_model()
    normalization_dict = load_normalization_dataset()

    st.markdown("### Pilih Metode Input")
    input_method = st.segmented_control(
        "Metode:",
        ["Input Teks Manual", "Upload File (CSV/Excel/TXT)"],
        selection_mode="single",
        default="Input Teks Manual",
        label_visibility="collapsed"
    )

    st.divider()

    if input_method == "Input Teks Manual":
        with st.form("text_input_form", enter_to_submit=False, clear_on_submit=True):
            st.markdown("### Input Teks")
            text_input = st.text_area(
                "Teks anda:",
                height=150,
                placeholder="Masukkan teks tentang ekonomi Indonesia...\nContoh: Ekonomi Indonesia menunjukkan pertumbuhan positif di tahun ini"
            )

            col1, col2, col3 = st.columns([1, 0.5, 1])
            with col2:
                reset_btn = st.form_submit_button("Reset", type="secondary", use_container_width=True)
            with col3:
                submit_btn = st.form_submit_button("Klasifikasi Sentimen", type="primary", use_container_width=True)

        if reset_btn:
            st.session_state.predictions = []
            st.session_state.preprocessing_steps = []
            st.rerun()

        if submit_btn:
            if text_input:
                valid, message = validasi_teks(text_input)
                if not valid:
                    st.error(f"❌ {message}")
                else:
                    with st.spinner("Sedang memproses..."):
                        result, prep_steps, error_msg = process_single_text(
                            text_input, normalization_dict, tokenizer, model, device
                        )
                        
                        st.session_state.preprocessing_steps = prep_steps
                        
                        if error_msg:
                            st.warning(f"⚠️ {error_msg}")
                        elif result:
                            # Save prediction
                            st.session_state.predictions.append({
                                'text': text_input[:100] + "...",
                                'label': result['label'],
                                'confidence': result['confidence'],
                                'timestamp': datetime.now()
                            })
                            
                            # Display result
                            display_single_result(result)
            else:
                st.warning("Silakan masukkan teks terlebih dahulu.")
        
        # Show preprocessing steps for text input
        if st.session_state.preprocessing_steps:
            display_preprocessing_steps(st.session_state.preprocessing_steps)
    
    else:  # Upload File method
        with st.form("file_upload_form"):
            st.markdown("### Upload File")
            uploaded_file = st.file_uploader(
                "Import File Excel/CSV/TXT",
                type=['txt', 'csv', 'xlsx'],
                help="Upload file berisi teks yang akan dianalisis. Untuk CSV/Excel, teks harus ada di kolom pertama."
            )
            
            col1, col2, col3 = st.columns([1, 0.5, 1])
            with col2:
                reset_btn = st.form_submit_button("Reset", type="secondary", use_container_width=True)
            with col3:
                submit_btn = st.form_submit_button("Klasifikasi Sentimen", type="primary", use_container_width=True)
        
        # Handle form submission
        if reset_btn:
            st.session_state.predictions = []
            st.session_state.preprocessing_steps = []
            st.rerun()
        
        if submit_btn:
            if uploaded_file:
                valid, message = validasi_file(uploaded_file)
                if not valid:
                    st.error(f"❌ {message}")
                else:
                    with st.spinner("Sedang memproses file..."):
                        # Read file
                        texts, error_msg = read_file_contents(uploaded_file)
                        
                        if error_msg:
                            st.error(f"❌ {error_msg}")
                        elif texts:
                            # Reset predictions for new batch
                            st.session_state.predictions = []
                            
                            # Process texts
                            results, skipped_count = process_file_texts(
                                texts, normalization_dict, tokenizer, model, device
                            )
                            
                            # Display results
                            display_batch_results(results, skipped_count, len(texts))
                        else:
                            st.error("❌ File tidak berisi teks yang dapat diproses")
            else:
                st.warning("Silakan upload file terlebih dahulu.")

    display_visualizations_and_download()

    if not st.session_state.predictions and not submit_btn:
        st.info("Pilih metode input dan tekan tombol 'Klasifikasi Sentimen' untuk memulai.")

def halaman_panduan():
    st.title("Panduan Penggunaan")

    tab1, tab2, tab3 = st.tabs(["Cara Penggunaan", "Interpretasi Hasil", "Detail Teknis"])

    with tab1:
        st.markdown("""
        ### Cara Menggunakan Aplikasi

        #### 1. Input Teks Langsung
        1. Buka halaman **Prediksi Sentimen**
        2. Masukkan teks pada kolom "Masukkan teks"
        3. **Pastikan tidak ada file yang diupload**
        4. Klik tombol **Klasifikasi Sentimen**
        5. Hasil prediksi akan ditampilkan

        #### 2. Upload File
        1. Buka halaman **Prediksi Sentimen**
        2. **Pastikan kolom teks kosong**
        3. Klik area upload dan pilih file (.txt, .csv, atau .xlsx)
        4. Klik tombol **Klasifikasi Sentimen**
        5. Sistem akan memproses semua teks dalam file
        6. Hasil ditampilkan dalam bentuk tabel

        #### ⚠️ Penting:
        - **Pilih salah satu metode input saja** (teks ATAU file)
        - Jika kedua input terisi, sistem akan menampilkan error
        - Tombol klasifikasi akan disabled jika tidak ada input atau ada konflik input

        #### 3. Download Hasil
        - Setelah melakukan prediksi, klik tombol **Download Hasil**
        - File CSV berisi hasil analisis akan terunduh
        """)

    with tab2:
        st.markdown("""
        ### Interpretasi Hasil Analisis

        #### Kategori Sentimen:
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.success("**Positif** 🟢")
            st.markdown("Pandangan optimis terhadap ekonomi, pertumbuhan, peningkatan")
        with col2:
            st.error("**Negatif** 🔴")
            st.markdown("Pandangan pesimis, kekhawatiran, penurunan ekonomi")

        st.markdown("""
        #### Confidence Score:
        - **>80%**: Prediksi sangat kuat dan dapat diandalkan
        - **60-80%**: Prediksi cukup kuat dengan tingkat keyakinan moderat
        - **<60%**: Prediksi kurang yakin, perlu ditinjau lebih lanjut

        #### Handling Data Kosong:
        - Teks yang setelah preprocessing menjadi kosong akan dilewati
        - Sistem akan memberikan peringatan untuk data yang tidak dapat diproses
        """)

    with tab3:
        st.markdown("""
        ### Detail Teknis

        #### Model yang Digunakan:
        - **Model**: IndoBERT Custom Architecture
        - **Base Model**: indobenchmark/indobert-base-p1
        - **Training Data**: Tweet ekonomi Indonesia (5000+ data)
        - **Akurasi**: ~94% pada dataset uji
        - **Max Token**: 128 tokens

        #### Tahapan Preprocessing:
        1. **Case folding**: Mengubah teks menjadi huruf kecil
        2. **HTML Unescape**: Decode HTML entities
        3. **Remove URLs**: Menghapus link website
        4. **Remove mentions/hashtags**: Menghapus @username dan #hashtag
        5. **Remove symbols & numbers**: Menghapus simbol dan angka
        6. **Whitespace cleaning**: Normalisasi spasi
        7. **Text normalization**: Normalisasi kata slang Indonesia

        #### Format File yang Didukung:
        - **TXT**: Plain text file
        - **CSV**: Comma-separated values (teks di kolom pertama)
        - **XLSX**: Excel file (teks di kolom pertama)
        - **Ukuran maksimal**: 10MB
        """)

if __name__ == "__main__":
    if st.session_state.page == 'utama':
        halaman_utama()
    elif st.session_state.page == 'prediksi':
        halaman_prediksi()
    elif st.session_state.page == 'panduan':
        halaman_panduan()