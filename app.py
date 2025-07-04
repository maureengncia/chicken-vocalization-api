import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import logging
from flask_cors import CORS
import json
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests dari Android

# Konfigurasi upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Buat folder upload jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class ChickenVocalizationAPI:
    def __init__(self, model_path, norm_path):
        """
        Inisialisasi API untuk klasifikasi vokalisasi ayam
        model_path: path ke file model .h5
        norm_path: path ke file normalisasi .npz
        """
        self.model_path = model_path
        self.norm_path = norm_path
        self.model = None
        self.norm_data = None
        
        # Konfigurasi MFCC (SESUAIKAN dengan training Anda)
        self.n_mfcc = 20
        self.hop_length = 256
        self.n_fft = 1024
        self.max_length = 250
        
        # Sample rate HARUS SAMA dengan training
        self.sample_rate = 16000  # Sesuai dengan yang Anda tetapkan
        
        # Label kategori (HARUS SAMA dengan training)
        self.categories = [
            'ayam betina marah',
            'ayam betina memanggil jantan', 
            'ketika ada ancaman',
            'setelah bertelur'
        ]
        
        # Load model dan normalisasi
        self.load_model_and_norm()
        
    def load_model_and_norm(self):
        """Load model dan data normalisasi"""
        try:
            logger.info("Loading model dan normalisasi...")
            
            # Load model
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"✅ Model loaded: {self.model_path}")
            
            # Load normalization data
            self.norm_data = np.load(self.norm_path)
            logger.info(f"✅ Normalisasi loaded: {self.norm_path}")
            logger.info(f"   Mean: {self.norm_data['mean']:.6f}")
            logger.info(f"   Std: {self.norm_data['std']:.6f}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise
    
    def extract_mfcc_features(self, audio_path, debug=False):
        """
        Ekstraksi fitur MFCC dari file audio
        PERSIS seperti training - NO EXTRA PREPROCESSING
        """
        try:
            # Load audio PERSIS seperti training
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if debug:
                logger.info(f"Audio loaded - Duration: {len(audio)/sr:.2f}s, SR: {sr}, Shape: {audio.shape}")
                logger.info(f"Audio stats - Min: {audio.min():.6f}, Max: {audio.max():.6f}, Mean: {audio.mean():.6f}")
            
            # Ekstraksi MFCC features - PERSIS seperti di training
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft
                # TIDAK ADA parameter tambahan - sama persis dengan training
            )
            
            if debug:
                logger.info(f"MFCC raw shape: {mfcc.shape}")
                logger.info(f"MFCC raw stats - Min: {mfcc.min():.6f}, Max: {mfcc.max():.6f}, Mean: {mfcc.mean():.6f}")
            
            # Padding atau truncate - PERSIS seperti training
            if mfcc.shape[1] < self.max_length:
                pad_width = self.max_length - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            elif mfcc.shape[1] > self.max_length:
                mfcc = mfcc[:, :self.max_length]
            
            if debug:
                logger.info(f"MFCC final shape: {mfcc.shape}")
                logger.info(f"MFCC final stats - Min: {mfcc.min():.6f}, Max: {mfcc.max():.6f}, Mean: {mfcc.mean():.6f}")
            
            return mfcc

        except Exception as e:
            logger.error(f"Error extracting MFCC: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def preprocess_for_prediction(self, mfcc_features, debug=False):
        """
        Preprocess MFCC untuk prediksi - PERSIS seperti training
        """
        try:
            # Reshape untuk CNN: (1, 20, 250, 1) - PERSIS seperti training
            mfcc_reshaped = mfcc_features.reshape(1, mfcc_features.shape[0], mfcc_features.shape[1], 1)
            
            if debug:
                logger.info(f"Reshaped to: {mfcc_reshaped.shape}")
                logger.info(f"Before norm - Min: {mfcc_reshaped.min():.6f}, Max: {mfcc_reshaped.max():.6f}")
            
            # Normalisasi PERSIS seperti training
            mfcc_normalized = (mfcc_reshaped - self.norm_data['mean']) / self.norm_data['std']
            
            if debug:
                logger.info(f"After norm - Min: {mfcc_normalized.min():.6f}, Max: {mfcc_normalized.max():.6f}")
                logger.info(f"Norm values - Mean: {self.norm_data['mean']:.6f}, Std: {self.norm_data['std']:.6f}")
            
            return mfcc_normalized.astype('float32')
            
        except Exception as e:
            logger.error(f"Error preprocessing: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def predict_audio(self, audio_path, debug=True):
        """
        Prediksi vokalisasi ayam dari file audio dengan DEBUG MODE
        """
        try:
            logger.info("=" * 50)
            logger.info("🔍 STARTING PREDICTION WITH DEBUG")
            logger.info("=" * 50)
            
            # 1. Ekstraksi MFCC dengan debug
            mfcc_features = self.extract_mfcc_features(audio_path, debug=debug)
            if mfcc_features is None:
                return None
            
            # 2. Preprocess untuk prediksi dengan debug
            processed_features = self.preprocess_for_prediction(mfcc_features, debug=debug)
            if processed_features is None:
                return None
            
            # 3. Prediksi
            logger.info("🤖 Running model prediction...")
            predictions = self.model.predict(processed_features, verbose=0)
            
            # 4. Get hasil prediksi
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            predicted_label = self.categories[predicted_class]
            
            # 5. Log detailed results
            logger.info("📊 PREDICTION RESULTS:")
            logger.info(f"   Top prediction: {predicted_label} ({confidence:.4f})")
            for i, category in enumerate(self.categories):
                prob = float(predictions[0][i])
                logger.info(f"   {category}: {prob:.4f}")
            
            # 6. Buat response dengan semua probabilitas
            result = {
                'predicted_class': predicted_label,
                'confidence': confidence,
                'all_predictions': {}
            }
            
            for i, category in enumerate(self.categories):
                result['all_predictions'][category] = float(predictions[0][i])
            
            logger.info("=" * 50)
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting: {str(e)}")
            logger.error(traceback.format_exc())
            return None

# Inisialisasi API
# GANTI PATH INI SESUAI LOKASI FILE MODEL ANDA
MODEL_PATH = 'chicken_cnn_model.h5'
NORM_PATH = 'chicken_cnn_model_norm.npz'

try:
    api = ChickenVocalizationAPI(MODEL_PATH, NORM_PATH)
    logger.info("🐔 API siap digunakan!")
except Exception as e:
    logger.error(f"❌ Gagal inisialisasi API: {str(e)}")
    api = None

def allowed_file(filename):
    """Check apakah file extension diizinkan"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    """Endpoint untuk test API"""
    return jsonify({
        'message': 'Chicken Vocalization Classification API',
        'status': 'active',
        'version': '1.0',
        'endpoints': {
            'predict': '/predict (POST)',
            'debug-predict': '/debug-predict (POST)',
            'health': '/health (GET)'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_loaded = api is not None and api.model is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'categories': api.categories if api else []
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint utama untuk prediksi vokalisasi ayam
    Menerima file audio dan mengembalikan hasil prediksi
    """
    try:
        # Check apakah API sudah terinisialisasi
        if api is None:
            return jsonify({
                'error': 'API not initialized',
                'message': 'Model tidak berhasil dimuat'
            }), 500
        
        # Check apakah ada file dalam request
        if 'audio' not in request.files:
            return jsonify({
                'error': 'No audio file',
                'message': 'Tidak ada file audio dalam request'
            }), 400
        
        file = request.files['audio']
        
        # Check apakah file dipilih
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Tidak ada file yang dipilih'
            }), 400
        
        # Check apakah file format diizinkan
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file format',
                'message': f'Format file tidak didukung. Gunakan: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Simpan file sementara
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        logger.info(f"File uploaded: {filename}")
        
        # Prediksi dengan debug mode
        result = api.predict_audio(temp_path, debug=True)
        
        # Hapus file sementara
        os.remove(temp_path)
        
        if result is None:
            return jsonify({
                'error': 'Prediction failed',
                'message': 'Gagal melakukan prediksi audio'
            }), 500
        
        # Return hasil prediksi
        response = {
            'success': True,
            'filename': filename,
            'prediction': result,
            'message': 'Prediksi berhasil'
        }
        
        logger.info(f"Prediction successful: {result['predicted_class']} ({result['confidence']:.4f})")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/debug-predict', methods=['POST'])
def debug_predict():
    """
    Endpoint debug untuk analisa preprocessing
    """
    try:
        if api is None:
            return jsonify({'error': 'API not initialized'}), 500
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Simpan file sementara
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        logger.info(f"🔍 DEBUG MODE - File uploaded: {filename}")
        
        # Prediksi dengan debug
        result = api.predict_audio(temp_path, debug=True)
        
        # Hapus file sementara
        os.remove(temp_path)
        
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        response = {
            'success': True,
            'filename': filename,
            'prediction': result,
            'debug_info': 'Check server logs for detailed preprocessing info'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in debug-predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'message': 'File terlalu besar. Maksimal 16MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 error"""
    return jsonify({
        'error': 'Not found',
        'message': 'Endpoint tidak ditemukan'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 error"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'Terjadi error pada server'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print("🚀 Starting Flask API Server...")
    print("📱 Endpoint untuk Android:")
    print("   POST /predict - Upload audio file untuk prediksi")
    print("   POST /debug-predict - Upload audio dengan debug mode")
    print("   GET /health - Check status API")
    print("   GET / - Info API")
    print(f"🔗 Server akan berjalan di port: {port}")
    print("📝 Pastikan file model sudah ada di folder yang sama!")
    
    # Jalankan server
    app.run(host='0.0.0.0', port=port, debug=debug_mode)