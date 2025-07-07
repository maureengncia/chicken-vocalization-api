# app.py - Flask API untuk Model Transformer Vokalisasi Ayam
# Compatible dengan kode training Transformer

import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds untuk reproducibility (sama dengan training)
def set_random_seeds(seed=42):
    """Set random seeds untuk hasil yang reproducible"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass

set_random_seeds(42)

# Inisialisasi Flask app
app = Flask(__name__)

# Konfigurasi
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}

# Global variables untuk model dan konfigurasi
model = None
normalization_values = None

# Label kategori vokalisasi (sama persis dengan training)
categories = [
    'ayam betina marah',
    'ayam betina memanggil jantan',
    'ketika ada ancaman',
    'setelah bertelur'
]

# Konfigurasi MFCC (sama persis dengan training code)
class MFCCConfig:
    def __init__(self):
        self.n_mfcc = 20
        self.hop_length = 512
        self.n_fft = 1024
        self.max_length = 250

mfcc_config = MFCCConfig()

def allowed_file(filename):
    """Check apakah file audio yang diupload valid"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_and_normalization():
    """Load model dan normalization values"""
    global model, normalization_values
    
    try:
        # Load model TensorFlow
        model_path = 'chicken_transformer_model.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} tidak ditemukan!")
            
        logger.info("Loading model Transformer...")
        model = tf.keras.models.load_model(model_path)
        logger.info("✅ Model berhasil dimuat!")
        
        # Load normalization values
        norm_path = 'chicken_transformer_model_norm.npz'
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Normalization file {norm_path} tidak ditemukan!")
            
        normalization_data = np.load(norm_path)
        normalization_values = {
            'mean': float(normalization_data['mean']),
            'std': float(normalization_data['std'])
        }
        logger.info("✅ Normalization values berhasil dimuat!")
        logger.info(f"   Mean: {normalization_values['mean']:.6f}")
        logger.info(f"   Std: {normalization_values['std']:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

def extract_mfcc_features(audio_path):
    """
    Ekstraksi fitur MFCC dari file audio untuk Transformer
    Sama persis dengan function di training code
    """
    try:
        # Load audio file dengan sample rate yang konsisten
        audio, sr = librosa.load(audio_path, sr=None)

        # Ekstraksi MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=mfcc_config.n_mfcc,
            hop_length=mfcc_config.hop_length,
            n_fft=mfcc_config.n_fft
        )

        # MFCC shape: (n_mfcc, max_length)
        # Untuk Transformer, kita butuh bentuk (max_length, n_mfcc) - sequence format

        # Padding atau truncate ke panjang yang sama untuk time frames
        if mfcc.shape[1] < mfcc_config.max_length:
            # Padding dengan zeros jika terlalu pendek
            pad_width = mfcc_config.max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        elif mfcc.shape[1] > mfcc_config.max_length:
            # Truncate jika terlalu panjang
            mfcc = mfcc[:, :mfcc_config.max_length]

        # Transpose untuk Transformer: (max_length, n_mfcc)
        # Ini membuat setiap time frame sebagai satu token dalam sequence
        mfcc = mfcc.T  # Shape: (max_length, n_mfcc)

        return mfcc

    except Exception as e:
        logger.error(f"Error processing audio {audio_path}: {e}")
        return None

def preprocess_audio(audio_path):
    """
    Preprocess audio file untuk prediksi
    Sama persis dengan pipeline training
    """
    try:
        # Extract MFCC (sama dengan training)
        mfcc_features = extract_mfcc_features(audio_path)
        
        if mfcc_features is None:
            return None
            
        # Convert ke float32 dan normalize (sama dengan training)
        mfcc_features = mfcc_features.astype('float32')
        mfcc_features = (mfcc_features - normalization_values['mean']) / normalization_values['std']
        
        # Add batch dimension: (1, 250, 20) untuk Transformer
        mfcc_features = np.expand_dims(mfcc_features, axis=0)
        
        logger.info(f"Preprocessed shape: {mfcc_features.shape}")
        return mfcc_features
        
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def home():
    """Endpoint untuk testing API"""
    return jsonify({
        'status': 'success',
        'message': 'Flask API untuk Klasifikasi Vokalisasi Ayam - Transformer Model',
        'model_loaded': model is not None,
        'categories': categories,
        'mfcc_config': {
            'n_mfcc': mfcc_config.n_mfcc,
            'hop_length': mfcc_config.hop_length,
            'n_fft': mfcc_config.n_fft,
            'max_length': mfcc_config.max_length
        },
        'tensorflow_version': tf.__version__,
        'normalization_loaded': normalization_values is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint utama untuk prediksi vokalisasi ayam
    """
    try:
        # Check apakah model sudah dimuat
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model Transformer belum dimuat!'
            }), 500
            
        if normalization_values is None:
            return jsonify({
                'status': 'error',
                'message': 'Normalization values belum dimuat!'
            }), 500
            
        # Check apakah ada file yang diupload
        if 'audio' not in request.files:
            return jsonify({
                'status': 'error', 
                'message': 'Tidak ada file audio yang diupload! Gunakan field "audio".'
            }), 400
            
        file = request.files['audio']
        
        # Check apakah file valid
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Nama file kosong!'
            }), 400
            
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': f'Format file tidak didukung! Gunakan: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
            
        logger.info(f"Processing file: {file.filename}")
        
        # Save file temporary
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
            
        try:
            # Preprocess audio (sama dengan training pipeline)
            logger.info("Extracting MFCC features...")
            features = preprocess_audio(temp_path)
            
            if features is None:
                return jsonify({
                    'status': 'error',
                    'message': 'Gagal memproses file audio! Pastikan file audio valid.'
                }), 400
                
            # Prediksi dengan model Transformer
            logger.info("Predicting with Transformer model...")
            predictions = model.predict(features, verbose=0)
            
            # Get hasil prediksi
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = categories[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get all predictions dengan confidence score
            all_predictions = []
            for idx, prob in enumerate(predictions[0]):
                all_predictions.append({
                    'class': categories[idx],
                    'confidence': float(prob)
                })
            
            # Sort by confidence (descending)
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"Prediction successful: {predicted_class} (confidence: {confidence:.4f})")
            
            # Response lengkap
            response = {
                'status': 'success',
                'prediction': {
                    'class': predicted_class,
                    'confidence': confidence,
                    'class_index': int(predicted_class_idx)
                },
                'all_predictions': all_predictions,
                'audio_info': {
                    'filename': file.filename,
                    'mfcc_shape': list(features.shape),
                    'preprocessing': 'MFCC -> Normalize -> Transformer'
                },
                'model_info': {
                    'type': 'Transformer',
                    'categories': categories,
                    'tensorflow_version': tf.__version__
                }
            }
            
            return jsonify(response)
            
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error dalam prediksi: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tensorflow_version': tf.__version__
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'status': 'error',
        'message': 'File terlalu besar! Maksimal 16MB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 error"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint tidak ditemukan!'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 error"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error!'
    }), 500

if __name__ == '__main__':
    # Load model saat startup
    logger.info("🚀 Starting Flask API...")
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    if load_model_and_normalization():
        logger.info("✅ API siap digunakan!")
        
        # Jalankan Flask app
        app.run(
            host='0.0.0.0',  # Untuk deployment
            port=int(os.environ.get('PORT', 5000)),  # Railway akan set PORT
            debug=False  # Set False untuk production
        )
    else:
        logger.error("❌ Gagal memuat model. Server tidak dapat dijalankan!")
