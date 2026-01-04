import streamlit as st
import pandas as pd
import numpy as np
import torch
import html
from transformers import AutoTokenizer, AutoConfig
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from io import BytesIO
import re
from model import IndoBERTCustom
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from wordcloud import WordCloud

WIB = timezone(timedelta(hours=7))

st.set_page_config(
    page_title="EconSens - Klasifikasi Sentimen Ekonomi Indonesia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'page' not in st.session_state:
    st.session_state.page = 'utama'
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'preprocessing_steps' not in st.session_state:
    st.session_state.preprocessing_steps = []
if 'upload_results' not in st.session_state:
    st.session_state.upload_results = None

with st.sidebar:
    with st.container(horizontal_alignment="center"):
        st.image("./logo.svg", width=100)
    st.divider()

    if st.button("🏠 Halaman Utama", width="stretch", type= "primary" if st.session_state.page == "utama" else "secondary"):
        st.session_state.page = "utama"
        st.rerun()

    if st.button("🔍 Prediksi Sentimen", width="stretch", type= "primary" if st.session_state.page == "prediksi" else "secondary"):
        st.session_state.page = "prediksi"
        st.rerun()

    if st.button("📖 Panduan Penggunaan", width="stretch", type= "primary" if st.session_state.page == "panduan" else "secondary"):
        st.session_state.page = 'panduan'
        st.rerun()

def halaman_utama():
    with st.container(horizontal_alignment="center"):
        st.image("./logo.svg", width=200)

    # st.markdown("### Aplikasi Klasifikasi Sentimen Perekonomian Indonesia")

    col1, col2 = st.columns(2, border=True)
    with col1:
        st.markdown("""
            #### Aplikasi Klasifikasi Sentimen Perekonomian Indonesia

            EconSens merupakan aplikasi klasifikasi sentimen yang dirancang khusus untuk
            menganalisis opini publik tentang perekonomian Indonesia dari media sosial
            menggunakan model IndoBERT yang telah dilatih. Aplikasi ini dapat memudahkan Anda
            dalam melakukan klasifikasi sentimen pada topik ekonomi Indonesia secara cepat
            dan akurat tanpa perlu mengeluarkan biaya apapun.
            """)
    with col2:
        st.markdown("""
            #### Model yang Digunakan untuk EconSens

            Model ini dilatih menggunakan dataset tweet ekonomi Indonesia dengan
            membandingkan 4 model: Logistic Regression, SVM, LSTM, dan IndoBERT,
            di mana IndoBERT menunjukkan performa terbaik. Model IndoBERT telah dilatih menggunakan
            lebih dari 5000 data dari media sosial X dengan akurasi yang didapatkan
            sekitar `94%`.
            """)

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
        MODEL_REPO = "rasyadmr/indobert_econsens"
        MODEL_FILENAME = "model.pth"
        TOKENIZER_NAME = "indobenchmark/indobert-base-p1"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            model_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILENAME
            )
            print(f"Model telah diunduh di: {model_path}")
        except Exception as e:
            st.error(f"❌ Gagal mengunduh file dari Hugging Face. Pastikan Repo ID dan Filename benar. Error: {str(e)}")
            return None, None, None

        try:
            model_data = torch.load(model_path, map_location=device)
        except Exception as e:
            st.error(f"❌ Tidak dapat membaca file model: {str(e)}")
            return None, None, None

        if 'model_config' not in model_data:
            st.error("❌ File model tidak memiliki konfigurasi 'model_config' yang diperlukan")
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

        if 'model_state_dict' in model_data:
            model.load_state_dict(model_data['model_state_dict'], strict=True)
        else:
            st.error("❌ File model tidak memiliki key 'model_state_dict'")
            return None, None, None

        model.eval()
        model = model.to(device)

        return tokenizer, model, device

    except Exception as e:
        st.error(f"❌ Error sistem memuat model: {str(e)}")
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
            st.warning(f"⚠️ Tidak dapat memuat dataset slang, menggunakan manual dict saja: {str(e)}")

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
        st.error(f"❌ Error memuat normalization dataset: {str(e)}")
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

    text = re.sub(r'@\S+|#\S+', '', text)
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
    if file is None:
        return False, "File tidak ditemukan"

    file_ext = file.name.split('.')[-1].lower()

    if file_ext != 'xlsx':
        return False, "Format file harus Excel (.xlsx)"
    if file.size > 10 * 1024 * 1024:
        return False, "Ukuran file terlalu besar (maksimal 10MB)"
    if file.size == 0:
        return False, "File kosong"
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

def create_wordcloud(text_data, title):
    stopwords = set([
        "yang", "di", "dan", "itu", "ini", "untuk", "dari", "ke", "akan",
        "pada", "juga", "dengan", "adalah", "karena", "bisa", "ada",
        "seperti", "saya", "tapi", "tidak", "ya", "yg", "gak", "kalo",
        "udah", "sdh", "aja", "n", "t", "saja", "kalau", "biar", "bikin",
        "bilang", "krn", "tp", "dgn", "sdh", "nih", "kok", "sih"
    ])

    combined_text = " ".join(text_data)
    if not combined_text:
        return None

    wordcloud = WordCloud(
        width=800,
        height=400,
        stopwords=stopwords,
        min_font_size=10,
        max_words=20,
        collocations=False,
        background_color="white"
    ).generate(combined_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=20)
    plt.tight_layout(pad=0)

    return fig

def process_single_text(text, normalization_dict, tokenizer, model, device):
    processed_text, prep_steps = preprocess_text(text, normalization_dict)

    if not processed_text or not processed_text.strip():
        return None, prep_steps, processed_text, "Teks tidak dapat diklasifikasi karena hasil preprocessing kosong"

    result = predict_sentiment_torch(processed_text, tokenizer, model, device)
    if result and not result.get('error'):
        return result, prep_steps, processed_text, None

    return None, prep_steps, processed_text, "Terjadi kesalahan saat prediksi"

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
            'text': text[:100] + "..." if len(text) > 100 else text,
            'full_text': text,
            'clean_text': processed_text,
            'label': result['label'],
            'confidence': result['confidence'],
            'timestamp': datetime.now(WIB)
        })
    return results, skipped_count

def read_file_contents(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, header=None)
        if df.empty:
            return None, "File Excel kosong"

        texts = df.iloc[:, 0].dropna().astype(str).tolist()

        if not texts:
            return None, "Tidak ada teks di kolom pertama file Excel"

        return texts, None
    except Exception as e:
        return None, f"Error membaca file Excel: {str(e)}"

def display_single_result(result):
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
    st.success(f"Selesai! Berhasil memproses {total_count - skipped_count} data.")
    if skipped_count > 0:
        st.warning(f"⚠️ {skipped_count} data dilewati karena teks kosong setelah preprocessing.")

    st.markdown("### Hasil Analisis")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, width="stretch", hide_index=True)

def display_preprocessing_steps(preprocessing_steps):
    with st.expander("Lihat Tahapan Preprocessing", expanded=False):
        for step, text in preprocessing_steps:
            st.text(f"{step}:")
            st.code(text if text else "(empty)", language=None)

def display_visualizations_and_download(input_method):
    if st.session_state.predictions:
        st.divider()
        st.markdown("### Visualisasi Analisis")

        data_vis = st.session_state.predictions
        try:
            fig_pie, fig_bar = create_visualization(data_vis)

            col1_vis, col2_vis = st.columns(2)
            with col1_vis:
                if fig_pie: st.plotly_chart(fig_pie, width="stretch")
                else: st.info("ℹ️ Data tidak cukup untuk Pie Chart.")
            with col2_vis:
                if fig_bar: st.plotly_chart(fig_bar, width="stretch")
                else: st.info("ℹ️ Data tidak cukup untuk Bar Chart.")
        except Exception as e:
            st.error(f"❌ Gagal membuat grafik: {str(e)}")

        if input_method == "Upload File Excel":
            all_texts = [p.get('clean_text', p['text']) for p in data_vis]
            pos_texts = [p.get('clean_text', p['text']) for p in data_vis if p['label'] == 'Positif']
            neg_texts = [p.get('clean_text', p['text']) for p in data_vis if p['label'] == 'Negatif']

            if all_texts:
                st.markdown("#### Semua Kata")
                fig_all = create_wordcloud(all_texts, "WordCloud - Keseluruhan")
                if fig_all: st.pyplot(fig_all)

            col_wc1, col_wc2 = st.columns(2)

            with col_wc1:
                if pos_texts:
                    st.markdown("#### Sentimen Positif")
                    fig_pos = create_wordcloud(pos_texts, "WordCloud - Positif")
                    if fig_pos: st.pyplot(fig_pos)
                else:
                    st.info("ℹ️ Tidak ada data sentimen positif.")

            with col_wc2:
                if neg_texts:
                    st.markdown("#### Sentimen Negatif")
                    fig_neg = create_wordcloud(neg_texts, "WordCloud - Negatif")
                    if fig_neg: st.pyplot(fig_neg)
                else:
                    st.info("ℹ️ Tidak ada data sentimen negatif.")

        st.divider()

        df_export = pd.DataFrame({
            'Teks': [p['full_text'] for p in st.session_state.predictions],
            'Sentimen': [p['label'] for p in st.session_state.predictions],
            'Tingkat Keyakinan (%)': [f"{p['confidence']*100:.2f}" for p in st.session_state.predictions],
            'Waktu Analisis': [p['timestamp'].strftime('%Y-%m-%d %H:%M:%S WIB') for p in st.session_state.predictions]
        })

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Hasil Analisis')

            worksheet = writer.sheets['Hasil Analisis']

            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

        excel_data = output.getvalue()

        _, col_download, _ = st.columns([1, 2, 1])
        with col_download:
            st.download_button(
                label="📥 Download Hasil (Excel)",
                data=excel_data,
                file_name=f"hasil_analisis_sentimen_{datetime.now(WIB).strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                type="primary",
                width="stretch"
            )

def reset_action():
    st.session_state.predictions = []
    st.session_state.preprocessing_steps = []
    st.session_state.upload_results = None

def halaman_prediksi():
    st.title("🔍 Prediksi Sentimen")

    tokenizer, model, device = load_model()
    normalization_dict = load_normalization_dataset()

    st.markdown("### Pilih Metode Input")
    input_method = st.segmented_control(
        "Metode:",
        ["Input Teks Manual", "Upload File Excel"],
        selection_mode="single",
        default="Input Teks Manual",
        label_visibility="collapsed",
        on_change=reset_action
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

            _, col_reset, col_submit = st.columns([1, 0.5, 1])
            with col_reset:
                reset_btn = st.form_submit_button("Reset", type="secondary", width="stretch")
            with col_submit:
                submit_btn = st.form_submit_button("Klasifikasi Sentimen", type="primary", width="stretch")

        if reset_btn:
            reset_action()
            st.rerun()

        if submit_btn:
            reset_action()
            if text_input:
                valid, message = validasi_teks(text_input)
                if not valid:
                    st.error(f"❌ {message}")
                else:
                    with st.spinner("🔄 Sedang memproses..."):
                        result, prep_steps, result_text, error_msg = process_single_text(
                            text_input, normalization_dict, tokenizer, model, device
                        )

                        st.session_state.preprocessing_steps = prep_steps

                        if error_msg:
                            st.warning(f"⚠️ {error_msg}")
                        elif result:
                            st.session_state.predictions.append({
                                'text': text_input[:100] + "...",
                                'full_text': text_input,
                                'clean_text': result_text,
                                'label': result['label'],
                                'confidence': result['confidence'],
                                'timestamp': datetime.now(WIB)
                            })

                            display_single_result(result)
            else:
                st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")

        if st.session_state.preprocessing_steps:
            display_preprocessing_steps(st.session_state.preprocessing_steps)

    else:
        with st.form("file_upload_form"):
            st.markdown("### Upload File Excel")
            uploaded_file = st.file_uploader(
                "Import File Excel",
                type=['xlsx'],
                help="Upload file Excel (.xlsx) berisi teks yang akan dianalisis. Teks harus ada di kolom pertama."
            )

            _, col_reset, col_submit = st.columns([1, 0.5, 1])
            with col_reset:
                reset_btn = st.form_submit_button("Reset", type="secondary", width="stretch")
            with col_submit:
                submit_btn = st.form_submit_button("Klasifikasi Sentimen", type="primary", width="stretch")

        if reset_btn:
            reset_action()
            st.rerun()

        if submit_btn:
            reset_action()
            if uploaded_file:
                valid, message = validasi_file(uploaded_file)
                if not valid:
                    st.error(f"❌ {message}")
                else:
                    with st.spinner("Sedang memproses file Excel..."):
                        texts, error_msg = read_file_contents(uploaded_file)

                        if error_msg:
                            st.error(f"❌ {error_msg}")
                        elif texts:
                            results, skipped_count = process_file_texts(
                                texts, normalization_dict, tokenizer, model, device
                            )

                            st.session_state.upload_results = {
                                'results': results,
                                'skipped': skipped_count,
                                'total': len(texts)
                            }
                        else:
                            st.error("❌ File Excel tidak berisi teks yang dapat diproses")
            else:
                st.warning("⚠️ Silakan upload file Excel terlebih dahulu.")

    if st.session_state.upload_results is not None:
        data = st.session_state.upload_results
        display_batch_results(data['results'], data['skipped'], data['total'])

    display_visualizations_and_download(input_method)

    if not st.session_state.predictions and not submit_btn:
        st.info("ℹ️ Pilih metode input, masukkan data, dan tekan tombol 'Klasifikasi Sentimen' untuk memulai.")

def halaman_panduan():
    st.title("📖 Panduan Penggunaan")

    tab1, tab2, tab3 = st.tabs(["Cara Penggunaan", "Interpretasi Hasil", "Detail Teknis"])

    with tab1:
        st.markdown("""
        ### 🆘 Cara Menggunakan Aplikasi
        """)

        with st.expander("✏️ Input Teks Langsung"):
            st.markdown("""
                1. Buka halaman **Prediksi Sentimen**
                2. Pilih metode **Input Teks Manual**
                3. Masukkan teks pada kolom yang tersedia
                4. Klik tombol **Klasifikasi Sentimen**
                5. Hasil prediksi akan ditampilkan secara langsung
            """)

        with st.expander("📁 Upload File Excel"):
            st.markdown("""
                1. Buka halaman **Prediksi Sentimen**
                2. Pilih metode **Upload File Excel**
                3. Klik area upload dan pilih file Excel (.xlsx)
                4. Pastikan teks yang akan dianalisis ada di **kolom pertama**
                5. Klik tombol **Klasifikasi Sentimen**
                6. Sistem akan memproses semua teks dalam file
                7. Hasil ditampilkan dalam bentuk tabel
            """)

        with st.expander("📄 Download Hasil"):
            st.markdown("""
                - Setelah melakukan prediksi, klik tombol **Download Hasil (Excel)**
                - File Excel akan berisi:
                - Teks asli
                - Hasil sentimen (Positif/Negatif)
                - Tingkat keyakinan dalam persentase
                - Waktu analisis (WIB)
            """)

    with tab2:
        st.markdown("""
        ### 📊 Interpretasi Hasil Analisis

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
        ### 🔧 Detail Teknis
        """)

        with st.expander("⚙️ Model yang digunakan"):
            st.markdown("""
                - **Model**: IndoBERT Custom Architecture
                - **Base Model**: indobenchmark/indobert-base-p1
                - **Training Data**: Tweet ekonomi Indonesia (5000+ data)
                - **Akurasi**: ~94% pada dataset uji
                - **Max Token**: 128 tokens
            """)

        with st.expander("🧹 Tahapan Preprocessing"):
            st.markdown("""
                1. **Case folding**: Mengubah teks menjadi huruf kecil
                2. **HTML Unescape**: Decode HTML entities
                3. **Remove URLs**: Menghapus link website
                4. **Remove mentions/hashtags**: Menghapus @username dan #hashtag
                5. **Remove symbols & numbers**: Menghapus simbol dan angka
                6. **Whitespace cleaning**: Normalisasi spasi
                7. **Text normalization**: Normalisasi kata slang Indonesia
            """)

        with st.expander("☑️ Format File yang Didukung"):
            st.markdown("""
                - **Format**: Excel (.xlsx)
                - **Struktur**: Teks harus berada di kolom pertama
                - **Ukuran maksimal**: 10MB
            """)

if __name__ == "__main__":
    if st.session_state.page == 'utama':
        halaman_utama()
    elif st.session_state.page == 'prediksi':
        halaman_prediksi()
    elif st.session_state.page == 'panduan':
        halaman_panduan()