import streamlit as st
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('indonesian'))

# Fungsi untuk memproses teks
def preprocess_text(text):
    # Tokenisasi
    tokens = word_tokenize(text.lower())
    # Menghapus stopwords dan tanda baca
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

# Daftar topik dan kata kunci terkait
topic_keywords = {
    "depresi": ["depresi", "sedih", "putus asa", "tidak berharga", "bunuh diri"],
    "kecemasan": ["cemas", "khawatir", "panik", "takut", "gelisah"],
    "stres": ["stres", "tertekan", "beban", "kewalahan", "burnout"],
    "tidur": ["tidur", "insomnia", "mimpi buruk", "kantuk", "lelah"],
    "perawatan_diri": ["self care", "perawatan diri", "relaksasi", "hobi", "istirahat"],
    "terapi": ["terapi", "konseling", "psikolog", "psikiater", "pengobatan"],
    "obat": ["obat", "medikasi", "antidepresan", "ssri", "efek samping"],
    "mindfulness": ["mindfulness", "meditasi", "kesadaran", "fokus", "tenang"],
    "hubungan": ["hubungan", "keluarga", "teman", "pasangan", "konflik"],
    "pekerjaan": ["kerja", "karir", "bos", "rekan kerja", "produktivitas"],
    "trauma": ["trauma", "ptsd", "kekerasan", "pelecehan", "pemulihan"],
    "adiksi": ["adiksi", "kecanduan", "alkohol", "narkoba", "rehabilitasi"],
    "bipolar": ["bipolar", "mania", "mood swing", "siklotimia", "stabilisator mood"],
    "skizofrenia": ["skizofrenia", "halusinasi", "delusi", "paranoia", "antipsikotik"],
    "ocd": ["ocd", "obsesif", "kompulsif", "ritual", "intrusive thoughts"],
    "makan": ["makan", "anoreksia", "bulimia", "binge eating", "body image"],
    "adhd": ["adhd", "hiperaktif", "inatensi", "impulsif", "konsentrasi"],
    "autism": ["autism", "asd", "asperger", "stimming", "sensory"],
    "perkembangan_anak": ["perkembangan anak", "milenstone", "tumbuh kembang", "parenting", "anak berkebutuhan khusus"],
    "lansia": ["lansia", "demensia", "alzheimer", "parkinson", "geriatri"],
    "bahagia": ["bahagia", "senang", "gembira", "ceria", "riang"],
    "sedih": ["sedih", "murung", "kecewa", "putus asa", "depresi"],
    "cemas": ["cemas", "khawatir", "gelisah", "takut", "panik"],
    "marah": ["marah", "kesal", "frustrasi", "jengkel", "emosi"],
    "stress": ["stress", "tertekan", "terbebani", "overwhelmed", "burnout"],
    "umum": ["kesehatan mental", "pikiran", "perasaan", "emosi", "wellbeing"]
}

# Daftar respons untuk setiap topik
responses = {
    "depresi": [
        "Depresi adalah kondisi kesehatan mental yang serius. Penting untuk mencari bantuan profesional jika Anda mengalami perasaan sedih atau putus asa yang terus-menerus.",
        "Beberapa cara untuk mengatasi depresi termasuk olahraga teratur, menjaga jadwal tidur yang sehat, dan berbicara dengan teman atau keluarga yang dipercaya.",
        "Ingatlah bahwa Anda tidak sendirian. Banyak orang mengalami depresi, dan ada banyak pilihan pengobatan yang efektif tersedia.",
        "Terapi Kognitif Perilaku (CBT) atau bentuk psikoterapi lainnya telah terbukti efektif dalam mengobati depresi.",
        "Jika Anda memiliki pikiran untuk menyakiti diri sendiri atau bunuh diri, segera hubungi layanan krisis atau layanan darurat."
    ],
    "kecemasan": [
        "Kecemasan adalah emosi manusia yang normal, tetapi ketika menjadi berlebihan, mungkin saatnya untuk mencari bantuan profesional.",
        "Latihan pernapasan dalam dan meditasi dapat membantu mengurangi gejala kecemasan.",
        "Cobalah untuk mengidentifikasi pemicu kecemasan Anda dan kembangkan strategi koping untuk masing-masing pemicu.",
        "Olahraga teratur sangat efektif dalam mengelola gejala kecemasan.",
        "Pertimbangkan untuk menyimpan jurnal untuk melacak gejala kecemasan Anda dan mengidentifikasi pola atau pemicu."
    ],
    "stres": [
        "Stres dapat dikelola melalui teknik relaksasi seperti relaksasi otot progresif atau pencitraan terpandu.",
        "Penting untuk menjaga keseimbangan antara pekerjaan dan kehidupan pribadi serta meluangkan waktu untuk aktivitas perawatan diri.",
        "Aktivitas fisik teratur dapat membantu mengurangi tingkat stres dan meningkatkan kesejahteraan secara keseluruhan.",
        "Jika stres berlangsung terus-menerus, pertimbangkan untuk berbicara dengan terapis atau konselor tentang teknik manajemen stres.",
        "Prioritaskan tugas-tugas Anda dan belajarlah untuk mengatakan tidak untuk menghindari rasa kewalahan."
    ],
    "tidur": [
        "Pola tidur yang konsisten sangat penting untuk kesehatan mental. Cobalah untuk tidur dan bangun pada waktu yang sama setiap hari.",
        "Ciptakan rutinitas sebelum tidur yang menenangkan untuk memberi sinyal pada tubuh Anda bahwa sudah waktunya untuk tidur.",
        "Hindari penggunaan layar elektronik setidaknya satu jam sebelum tidur, karena cahaya biru dapat mengganggu siklus tidur Anda.",
        "Jika Anda mengalami masalah tidur yang terus-menerus, bicarakan dengan dokter Anda tentang kemungkinan penyebab atau pilihan pengobatan.",
        "Pertimbangkan untuk menggunakan suara white noise atau suara menenangkan untuk menciptakan lingkungan yang lebih kondusif untuk tidur."
    ],
    "perawatan_diri": [
        "Perawatan diri sangat penting untuk menjaga kesehatan mental yang baik. Ini bisa termasuk aktivitas seperti membaca, mandi, atau melakukan hobi.",
        "Pastikan untuk menjadwalkan waktu rutin untuk aktivitas yang Anda nikmati dan yang membantu Anda rileks.",
        "Perawatan diri fisik, termasuk olahraga teratur dan diet sehat, dapat berdampak signifikan pada kesejahteraan mental Anda.",
        "Jangan lupakan perawatan diri sosial. Menjaga hubungan yang sehat dan koneksi sosial penting untuk kesehatan mental.",
        "Mindfulness dan meditasi bisa menjadi alat perawatan diri yang kuat untuk mengelola stres dan meningkatkan kesejahteraan secara keseluruhan."
    ],
    "terapi": [
        "Terapi bisa menjadi alat yang berharga untuk mengelola kesehatan mental. Ada banyak jenis terapi, termasuk Terapi Kognitif Perilaku (CBT), terapi psikodinamik, dan lainnya.",
        "Jika Anda mempertimbangkan terapi, riset berbagai pendekatan untuk menemukan yang mungkin paling cocok untuk Anda.",
        "Ingat, mungkin butuh waktu untuk menemukan terapis yang membuat Anda nyaman. Tidak apa-apa untuk mencoba beberapa sebelum memutuskan.",
        "Pilihan terapi online sekarang tersedia secara luas dan bisa menjadi alternatif yang nyaman dari sesi tatap muka tradisional.",
        "Terapi kelompok bisa bermanfaat bagi beberapa orang, menyediakan lingkungan yang mendukung dan kesempatan untuk terhubung dengan orang lain yang menghadapi tantangan serupa."
    ],
    "obat": [
        "Obat bisa menjadi pengobatan yang efektif untuk banyak kondisi kesehatan mental ketika diresepkan dan dipantau oleh profesional kesehatan.",
        "Penting untuk mengonsumsi obat yang diresepkan sesuai petunjuk dan mengomunikasikan efek samping apa pun kepada dokter Anda.",
        "Obat sering kali bekerja paling baik ketika dikombinasikan dengan terapi atau bentuk pengobatan lainnya.",
        "Jangan pernah berhenti mengonsumsi obat yang diresepkan tanpa berkonsultasi dengan dokter Anda terlebih dahulu.",
        "Jika Anda mempertimbangkan pengobatan, diskusikan potensi manfaat dan risiko dengan psikiater atau dokter umum Anda."
    ],
    "mindfulness": [
        "Mindfulness melibatkan fokus pada momen sekarang dan bisa menjadi alat yang kuat untuk mengelola stres dan kecemasan.",
        "Praktik mindfulness secara teratur dapat membantu meningkatkan regulasi emosi dan kesejahteraan secara keseluruhan.",
        "Ada banyak cara untuk mempraktikkan mindfulness, termasuk meditasi, pernapasan mindful, dan pemindaian tubuh.",
        "Mindfulness dapat diintegrasikan ke dalam aktivitas sehari-hari, seperti makan, berjalan, atau bahkan mencuci piring.",
        "Aplikasi dan sumber daya online dapat membantu Anda memulai praktik mindfulness atau memperdalam praktik yang sudah ada."
    ],
    "hubungan": [
        "Hubungan yang sehat sangat penting untuk kesejahteraan mental. Komunikasi terbuka dan jujur adalah kunci dalam semua jenis hubungan.",
        "Jika Anda mengalami konflik dalam hubungan, cobalah untuk mendengarkan secara aktif dan memahami sudut pandang orang lain.",
        "Menetapkan batasan yang sehat dalam hubungan dapat membantu mengurangi stres dan meningkatkan kesejahteraan emosional.",
        "Jika Anda mengalami masalah hubungan yang terus-menerus, terapi pasangan atau keluarga mungkin bisa membantu.",
        "Ingat untuk merawat diri sendiri bahkan ketika fokus pada hubungan dengan orang lain."
    ],
    "pekerjaan": [
        "Stres terkait pekerjaan adalah umum, tetapi ada cara untuk mengelolanya. Cobalah untuk memprioritaskan tugas dan kelola waktu Anda secara efektif.",
        "Jika Anda merasa kewalahan di tempat kerja, bicarakan dengan supervisor Anda tentang cara untuk mengelola beban kerja Anda.",
        "Pastikan untuk mengambil istirahat teratur selama hari kerja dan gunakan waktu istirahat untuk relaksasi atau aktivitas yang menyegarkan.",
        "Menjaga keseimbangan antara pekerjaan dan kehidupan pribadi sangat penting untuk kesehatan mental. Tetapkan batasan yang jelas antara waktu kerja dan waktu pribadi.",
        "Jika stres atau kecemasan terkait pekerjaan terus berlanjut, pertimbangkan untuk berbicara dengan profesional kesehatan mental atau konselor karir."
    ],
    "trauma": [
        "Trauma dapat memiliki dampak jangka panjang pada kesehatan mental. Penting untuk mencari bantuan profesional jika Anda mengalami gejala PTSD.",
        "Terapi EMDR (Eye Movement Desensitization and Reprocessing) telah terbukti efektif dalam mengobati trauma.",
        "Praktik grounding dapat membantu ketika Anda merasa overwhelmed oleh ingatan traumatis.",
        "Membangun jaringan dukungan yang kuat adalah bagian penting dari pemulihan trauma.",
        "Ingat bahwa pemulihan dari trauma adalah proses, dan setiap orang memiliki jalan pemulihan yang berbeda."
    ],
    "adiksi": [
        "Adiksi adalah kondisi kesehatan yang kompleks yang mempengaruhi otak dan perilaku. Penting untuk mencari bantuan profesional untuk mengatasi adiksi.",
        "Program 12 langkah seperti AA atau NA telah membantu banyak orang dalam pemulihan dari adiksi.",
        "Dukungan keluarga dan teman dapat sangat membantu dalam proses pemulihan dari adiksi.",
        "Terapi perilaku kognitif (CBT) dan terapi motivasi sering digunakan dalam pengobatan adiksi.",
        "Ingat bahwa kambuh adalah bagian umum dari proses pemulihan dan bukan berarti kegagalan."
    ],
    "bipolar": [
        "Gangguan bipolar ditandai oleh perubahan mood yang ekstrem. Penting untuk mendapatkan diagnosis dan perawatan yang tepat.",
        "Stabilisator mood sering digunakan dalam pengobatan gangguan bipolar untuk membantu mengendalikan perubahan mood.",
        "Mengenali tanda-tanda awal episode manik atau depresif dapat membantu dalam manajemen kondisi.",
        "Rutinitas yang konsisten, termasuk jadwal tidur yang teratur, sangat penting dalam mengelola gangguan bipolar.",
        "Psikoedukasi dapat membantu Anda dan orang-orang terdekat Anda untuk lebih memahami dan mengelola kondisi ini."
    ],
    "skizofrenia": [
        "Skizofrenia adalah gangguan mental kronis yang mempengaruhi cara seseorang berpikir, merasa, dan berperilaku. Pengobatan yang tepat sangat penting.",
        "Obat antipsikotik adalah komponen utama dalam pengobatan skizofrenia dan dapat membantu mengelola gejala seperti halusinasi dan delusi.",
        "Terapi psikososial, seperti terapi perilaku kognitif (CBT), dapat membantu individu dengan skizofrenia mengelola gejala mereka dan meningkatkan fungsi sehari-hari.",
        "Dukungan keluarga dan pendidikan tentang kondisi ini sangat penting dalam manajemen skizofrenia jangka panjang.",
        "Mengenali tanda-tanda awal kambuh dapat membantu dalam pencegahan dan intervensi dini."
    ],
    "ocd": [
        "OCD (Obsessive-Compulsive Disorder) ditandai oleh pikiran yang mengganggu (obsesi) dan perilaku berulang (kompulsi). Pengobatan dapat sangat membantu dalam mengelola gejala.",
        "Terapi Perilaku Kognitif (CBT), khususnya Exposure and Response Prevention (ERP), adalah pengobatan pilihan untuk OCD.",
        "Obat-obatan, seperti SSRI, juga dapat digunakan dalam pengobatan OCD, terutama dalam kasus yang lebih parah.",
        "Teknik mindfulness dapat membantu individu dengan OCD untuk lebih sadar akan pikiran mereka tanpa bereaksi berlebihan.",
        "Penting untuk tidak memperkuat perilaku kompulsif, meskipun ini bisa sangat menantang. Seekinglah bantuan profesional untuk strategi manajemen yang efektif."
    ],
    "makan": [
        "Gangguan makan, seperti anoreksia, bulimia, dan binge eating disorder, adalah kondisi serius yang memerlukan perawatan profesional.",
        "Pengobatan untuk gangguan makan biasanya melibatkan tim multidisiplin, termasuk dokter, psikiater, psikolog, dan ahli gizi.",
        "Terapi perilaku kognitif (CBT) dan terapi interpersonal (IPT) telah terbukti efektif dalam pengobatan beberapa gangguan makan.",
        "Pemulihan dari gangguan makan seringkali melibatkan perubahan pola pikir tentang makanan, tubuh, dan self-image.",
        "Dukungan keluarga dan teman dapat memainkan peran penting dalam proses pemulihan dari gangguan makan."
    ],
    "adhd": [
        "ADHD (Attention Deficit Hyperactivity Disorder) adalah gangguan neurodevelopmental yang mempengaruhi konsentrasi, impulsivitas, dan tingkat aktivitas.",
        "Pengobatan untuk ADHD biasanya melibatkan kombinasi dari obat-obatan (seperti stimulan) dan terapi perilaku.",
        "Strategi manajemen waktu dan organisasi dapat sangat membantu individu dengan ADHD dalam mengelola gejala mereka.",
        "Olahraga teratur dan tidur yang cukup dapat membantu mengurangi gejala ADHD.",
        "Penting untuk fokus pada kekuatan dan bakat individu dengan ADHD, bukan hanya tantangan mereka."
    ],
    "autism": [
        "Autism Spectrum Disorder (ASD) adalah gangguan neurodevelopmental yang mempengaruhi komunikasi, interaksi sosial, dan perilaku.",
        "Intervensi dini, seperti terapi bicara dan okupasi, dapat sangat membantu anak-anak dengan ASD dalam mengembangkan keterampilan penting.",
        "Setiap individu dengan ASD unik, dengan kekuatan dan tantangan mereka sendiri. Penting untuk mengadopsi pendekatan yang dipersonalisasi.",
        "Dukungan sensori dan strategi regulasi diri dapat membantu individu dengan ASD mengelola overload sensorik.",
        "Banyak orang dengan ASD memiliki minat khusus yang intens, yang bisa menjadi sumber kekuatan dan kesenangan."
    ],
    "perkembangan_anak": [
        "Setiap anak berkembang dengan kecepatan mereka sendiri, tetapi ada tonggak perkembangan umum yang bisa dijadikan panduan.",
        "Stimulasi dini melalui permainan, membaca, dan interaksi sosial sangat penting untuk perkembangan kognitif dan emosional anak.",
        "Pola asuh yang positif dan konsisten dapat membantu anak mengembangkan keterampilan regulasi emosi yang sehat.",
        "Jika Anda memiliki kekhawatiran tentang perkembangan anak Anda, bicarakan dengan pediatri atau ahli perkembangan anak.",
        "Ingat bahwa setiap anak unik dan mungkin tidak selalu mengikuti jadwal perkembangan 'standar'."
    ],
    "lansia": [
        "Kesehatan mental pada lansia sama pentingnya dengan kesehatan fisik. Depresi dan kecemasan bukan bagian normal dari penuaan.",
        "Aktivitas sosial dan keterlibatan dalam komunitas dapat membantu menjaga kesehatan kognitif dan emosional pada lansia.",
        "Olahraga teratur, bahkan dalam bentuk yang ringan seperti berjalan, dapat membantu menjaga kesehatan fisik dan mental pada lansia.",
        "Jika Anda melihat perubahan mendadak dalam perilaku atau kemampuan kognitif lansia, segera cari bantuan medis karena ini bisa menjadi tanda masalah kesehatan yang serius.",
        "Perawatan diri dan rutinitas yang konsisten dapat membantu lansia mempertahankan kemandirian dan kualitas hidup mereka."
    ],
    "bahagia": [
        "Senang mendengar Anda merasa bahagia! Perasaan positif seperti ini sangat baik untuk kesehatan mental Anda.",
        "Kebahagiaan adalah emosi yang indah. Cobalah untuk mengenali hal-hal yang membuat Anda bahagia dan lakukan lebih sering.",
        "Bagus sekali! Merayakan momen-momen bahagia, sekecil apapun, dapat membantu meningkatkan kesejahteraan mental Anda."
    ],
    "sedih": [
        "Saya mengerti Anda sedang merasa sedih. Ingatlah bahwa perasaan ini normal dan akan berlalu.",
        "Kesedihan adalah emosi yang wajar. Jika Anda merasa terlalu terbebani, jangan ragu untuk berbicara dengan seseorang yang Anda percaya.",
        "Dalam masa-masa sulit, penting untuk merawat diri sendiri. Cobalah melakukan aktivitas yang Anda sukai atau yang menenangkan."
    ],
    "cemas": [
        "Kecemasan bisa sangat mengganggu. Cobalah teknik pernapasan dalam atau meditasi singkat untuk menenangkan diri.",
        "Perasaan cemas adalah normal, tapi jika terlalu intens, pertimbangkan untuk berbicara dengan profesional kesehatan mental.",
        "Saat cemas, cobalah fokus pada hal-hal yang bisa Anda kontrol. Membuat daftar atau rencana sederhana bisa membantu."
    ],
    "marah": [
        "Kemarahan adalah emosi yang kuat. Cobalah untuk mengambil jeda sejenak dan menarik napas dalam-dalam sebelum bereaksi.",
        "Penting untuk mengenali pemicu kemarahan Anda. Dengan memahaminya, Anda bisa lebih baik dalam mengelola emosi ini.",
        "Ekspresi kemarahan yang sehat itu penting. Cobalah menulis jurnal atau berolahraga untuk menyalurkan energi ini."
    ],
    "stress": [
        "Stress dapat sangat melelahkan. Pastikan Anda meluangkan waktu untuk istirahat dan melakukan hal-hal yang Anda nikmati.",
        "Mengelola stress penting untuk kesehatan mental. Cobalah teknik relaksasi atau mindfulness untuk membantu meredakan stress.",
        "Jika Anda merasa stress berkepanjangan, jangan ragu untuk mencari dukungan dari teman, keluarga, atau profesional."
    ],
    "umum": [
        "Kesehatan mental adalah bagian penting dari kesehatan secara keseluruhan. Jika Anda merasa kesulitan, jangan ragu untuk mencari bantuan profesional.",
        "Merawat kesehatan mental sama pentingnya dengan merawat kesehatan fisik. Pastikan Anda memprioritaskan self-care dalam rutinitas harian Anda.",
        "Setiap orang memiliki tantangan kesehatan mental yang berbeda. Yang terpenting adalah mengenali kebutuhan Anda dan mencari dukungan yang sesuai.",
        "Perawatan diri sangat penting untuk kesehatan mental. Pastikan Anda cukup tidur, makan dengan baik, dan berolahraga secara teratur.",
        "Jika Anda merasa stres atau cemas, cobalah teknik relaksasi seperti pernapasan dalam atau meditasi.",
        "Berbicara dengan seseorang yang Anda percaya tentang perasaan Anda bisa sangat membantu.",
        "Ingat, tidak apa-apa untuk tidak merasa baik-baik saja. Mencari bantuan adalah tanda kekuatan, bukan kelemahan."
    ]
}

# Fungsi untuk mendapatkan respons berdasarkan similaritas
def get_response(user_input):
    preprocessed_input = preprocess_text(user_input)
    
    all_keywords = ' '.join([' '.join(keywords) for keywords in topic_keywords.values()])
    corpus = [all_keywords, preprocessed_input]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()
    
    if cosine_similarities[0] < 0.1:
        return random.choice(responses["umum"])
    
    topic_similarities = {
        topic: cosine_similarity(vectorizer.transform([' '.join(keywords)]), tfidf_matrix[1:2]).flatten()[0]
        for topic, keywords in topic_keywords.items()
    }
    
    best_topic = max(topic_similarities, key=topic_similarities.get)
    
    return random.choice(responses[best_topic])

# Fungsi untuk menyimpan riwayat chat
def save_chat_history(chat_history):
    df = pd.DataFrame([(item['role'], item['message']) for item in chat_history], columns=['Role', 'Message'])
    df.to_csv('chat_history.csv', index=False)

# Fungsi untuk memuat riwayat chat
def load_chat_history():
    try:
        df = pd.read_csv('chat_history.csv')
        return [{'role': row['Role'], 'message': row['Message'], 'id': str(uuid.uuid4())} for _, row in df.iterrows()]
    except FileNotFoundError:
        return []

# Fungsi untuk menganalisis sentimen (sederhana)
def analyze_sentiment(text):
    positive_words = set(['bahagia', 'senang', 'gembira', 'positif', 'baik'])
    negative_words = set(['sedih', 'marah', 'kecewa', 'negatif', 'buruk'])
    
    words = set(preprocess_text(text).split())
    
    positive_score = len(words.intersection(positive_words))
    negative_score = len(words.intersection(negative_words))
    
    if positive_score > negative_score:
        return 'Positif'
    elif negative_score > positive_score:
        return 'Negatif'
    else:
        return 'Netral'

# Fungsi untuk membuat grafik analisis sentimen
def plot_sentiment_analysis(chat_history):
    sentiments = [analyze_sentiment(item['message']) for item in chat_history if item['role'] == 'User']
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Analisis Sentimen Percakapan')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    st.pyplot(plt)

# Fungsi untuk mereset riwayat chat
def reset_chat_history():
    st.session_state.chat_history = []
    if os.path.exists('chat_history.csv'):
        os.remove('chat_history.csv')

# New function to get user's feeling
def get_user_feeling():
    feeling = st.text_input("Bagaimana perasaan Anda hari ini ?")
    if feeling:
        response = f"Saya mengerti bahwa Anda merasa {feeling} hari ini. Bisa ceritakan lebih lanjut?"
        st.session_state.chat_history.append({'role': 'User', 'message': feeling, 'id': str(uuid.uuid4())})
        st.session_state.chat_history.append({'role': 'Bot', 'message': response, 'id': str(uuid.uuid4())})
        save_chat_history(st.session_state.chat_history)
        st.session_state.conversation_stage = 'random_chat'
        st.experimental_rerun()
        
# Fungsi utama Streamlit
def main():
    st.title("SedulurRasa")
    st.write("Selamat datang di Chatbot Kesehatan Mental SedulurRasa. Silakan ajukan pertanyaan atau ungkapkan perasaan Anda tentang kesehatan mental, dan saya akan mencoba membantu.")

    # CSS untuk tampilan chat
    st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        background-color: black;
        border-radius: 10px;
    }
    .user-message {
        align-self: flex-end;
        background-color: #DCF8C6;
        color: black;
        padding: 10px;
        border-radius: 10px;
        max-width: 80%;
        word-wrap: break-word;
        margin-left: 20%;
        margin-right: 0;
    }
    .bot-message {
        align-self: flex-start;
        background-color: #FFFFFF;
        color: black;
        padding: 10px;
        border-radius: 10px;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
        margin-right: 20%;
        margin-left: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    if 'conversation_stage' not in st.session_state:
        st.session_state.conversation_stage = 'ask_name'

    # Display chat history
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for item in st.session_state.chat_history:
            if item['role'] == "User":
                chat_html += f'<div class="user-message"><strong>Anda : </strong> {item["message"]}</div>'
            else:
                chat_html += f'<div class="bot-message"><strong>SedulurRasa : </strong> {item["message"]}</div>'
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    # Conversation flow
    if st.session_state.conversation_stage == 'ask_name':
        name_input = st.text_input("Siapa nama Anda ? ", key="user_name_input")
        if name_input:
            st.session_state.user_name = name_input
            greeting_message = f"Halo, {st.session_state.user_name}! Salam kenal, aku Sedulurmu, siap mendengarkan."
            st.session_state.chat_history.append({'role': 'Bot', 'message': greeting_message, 'id': str(uuid.uuid4())})
            save_chat_history(st.session_state.chat_history)
            st.session_state.conversation_stage = 'ask_feeling'
            st.experimental_rerun()

    elif st.session_state.conversation_stage == 'ask_feeling':
        get_user_feeling()

    elif st.session_state.conversation_stage == 'random_chat':
        user_input = st.text_input("", key="user_input")
        
        if user_input:
            response = get_response(user_input)
            st.session_state.chat_history.append({'role': 'User', 'message': user_input, 'id': str(uuid.uuid4())})
            st.session_state.chat_history.append({'role': 'Bot', 'message': response, 'id': str(uuid.uuid4())})
            save_chat_history(st.session_state.chat_history)
            st.experimental_rerun()

        col1, col2 = st.columns([1.5, 0.5])

        with col1:
            if st.button("Reset Riwayat Chat"):
                reset_chat_history()
                st.session_state.conversation_stage = 'ask_name'
                st.session_state.user_name = None
                st.success("Riwayat chat telah dihapus!")
                st.experimental_rerun()

        with col2:
            if st.button("Unduh Riwayat Chat"):
                df = pd.DataFrame([(item['role'], item['message']) for item in st.session_state.chat_history], columns=['Role', 'Message'])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Unduh sebagai CSV",
                    data=csv,
                    file_name="chat_history.csv",
                    mime="text/csv"
                )
        
        st.markdown("<hr>", unsafe_allow_html=True)

        # Tombol untuk menganalisis sentimen
        if st.button("Analisis Sentimen"):
            plot_sentiment_analysis(st.session_state.chat_history)
       

    st.markdown("---")  # Garis pemisah
    st.write("Catatan: Chatbot ini hanya memberikan informasi umum dan bukan pengganti konsultasi dengan profesional kesehatan mental. Jika Anda memiliki masalah kesehatan mental yang serius, silakan hubungi profesional kesehatan atau layanan darurat.")

if __name__ == "__main__":
    main()