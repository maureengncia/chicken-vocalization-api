import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import time

# Inisialisasi Flask app
app = Flask(__name__)

# Konfigurasi
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB file
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

# Global variables untuk model dan normalization
model = None
norm_mean = None
norm_std = None
categories = [
    'ayam betina marah',
    'ayam betina memanggil jantan', 
    'ketika ada ancaman',
    'setelah bertelur'
]

# Konfigurasi MFCC (sama dengan training)
N_MFCC = 20
HOP_LENGTH = 512
N_FFT = 1024
MAX_LENGTH = 250
PATCH_SIZE = 10

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_mfcc_features(audio_path):
    """
    Ekstraksi fitur MFCC dari file audio (sama dengan training)
    """
    try:
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Ekstraksi MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT
        )
        
        # Padding atau truncate
        if mfcc.shape[1] < MAX_LENGTH:
            pad_width = MAX_LENGTH - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        elif mfcc.shape[1] > MAX_LENGTH:
            mfcc = mfcc[:, :MAX_LENGTH]
        
        # Transpose untuk Transformer: (max_length, n_mfcc)
        mfcc = mfcc.T  # Shape: (250, 20)
        
        return mfcc
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def preprocess_audio(mfcc_features):
    """
    Preprocess MFCC features untuk prediksi
    """
    try:
        # Reshape untuk batch: (1, max_length, n_mfcc)
        mfcc_batch = np.expand_dims(mfcc_features, axis=0)
        
        # Normalisasi dengan mean/std dari training
        mfcc_normalized = (mfcc_batch - norm_mean) / norm_std
        
        return mfcc_normalized.astype('float32')
        
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return None

def load_model_and_normalization():
    """
    Load model dan normalization values
    """
    global model, norm_mean, norm_std
    
    try:
        # Load model
        print("Loading Transformer model...")
        model = tf.keras.models.load_model('chicken_transformer_model.h5')
        print("✅ Model loaded successfully!")
        
        # Load normalization values
        print("Loading normalization values...")
        norm_data = np.load('chicken_transformer_model_norm.npz')
        norm_mean = norm_data['mean']
        norm_std = norm_data['std']
        print(f"✅ Normalization loaded - Mean: {norm_mean:.6f}, Std: {norm_std:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@app.route('/', methods=['GET'])
def home():
    """
    Endpoint untuk testing server
    """
    return jsonify({
        'status': 'success',
        'message': 'Chicken Vocalization Transformer API is running!',
        'model_loaded': model is not None,
        'categories': categories
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk prediksi vokalisasi ayam
    """
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
        
        # Check if file is in request
        if 'audio' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No audio file in request'
            }), 400
        
        file = request.files['audio']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': f'File type not allowed. Use: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # Extract MFCC features
        print(f"Processing audio: {filename}")
        mfcc_features = extract_mfcc_features(temp_path)
        
        if mfcc_features is None:
            # Cleanup
            os.remove(temp_path)
            os.rmdir(temp_dir)
            return jsonify({
                'status': 'error',
                'message': 'Failed to extract MFCC features'
            }), 400
        
        # Preprocess untuk prediksi
        mfcc_processed = preprocess_audio(mfcc_features)
        
        if mfcc_processed is None:
            # Cleanup
            os.remove(temp_path)
            os.rmdir(temp_dir)
            return jsonify({
                'status': 'error',
                'message': 'Failed to preprocess audio'
            }), 400
        
        # Prediksi dengan model Transformer
        print("Running prediction...")
        predictions = model.predict(mfcc_processed, verbose=0)
        
        # Get hasil prediksi
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        predicted_label = categories[predicted_class]
        
        # Get semua probabilitas
        all_predictions = []
        for i, prob in enumerate(predictions[0]):
            all_predictions.append({
                'class': categories[i],
                'probability': float(prob)
            })
        
        # Sort by probability (descending)
        all_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Cleanup temporary file
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return hasil
        return jsonify({
            'status': 'success',
            'prediction': {
                'class': predicted_label,
                'confidence': confidence,
                'class_index': int(predicted_class)
            },
            'all_predictions': all_predictions,
            'processing_time': round(processing_time, 3),
            'audio_info': {
                'filename': filename,
                'mfcc_shape': mfcc_features.shape
            }
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tensorflow_version': tf.__version__
    })

if __name__ == '__main__':
    print("🐔 STARTING CHICKEN VOCALIZATION TRANSFORMER API 🐔")
    print("="*60)
    
    # Load model dan normalization
    if load_model_and_normalization():
        print("🚀 Starting Flask server...")
        print("📱 API Endpoints:")
        print("   GET  /        - Home & API info")
        print("   POST /predict - Audio prediction")
        print("   GET  /health  - Health check")
        print("="*60)
        
        # Start server
        app.run(
            host='0.0.0.0',  # Accept connections from any IP
            port=5000,       # Port 5000
            debug=False      # Production mode
        )
    else:
        print("❌ Failed to start server - Model loading failed")
        print("📁 Make sure these files exist:")
        print("   - chicken_transformer_model.h5")
        print("   - chicken_transformer_model_norm.npz")
