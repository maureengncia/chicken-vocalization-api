from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
from werkzeug.utils import secure_filename
import tempfile
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables
model = None
norm_mean = None
norm_std = None

# Konfigurasi MFCC - SAMA dengan training
N_MFCC = 20
HOP_LENGTH = 512
N_FFT = 1024
MAX_LENGTH = 250

# Categories - SAMA dengan training
categories = [
    'ayam betina marah',
    'ayam betina memanggil jantan',
    'ketika ada ancaman',
    'setelah bertelur'
]

def extract_mfcc_features(audio_path):
    """Extract MFCC features - SAMA dengan training"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT
        )
        
        # Padding/truncate
        if mfcc.shape[1] < MAX_LENGTH:
            pad_width = MAX_LENGTH - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        elif mfcc.shape[1] > MAX_LENGTH:
            mfcc = mfcc[:, :MAX_LENGTH]
        
        # Transpose untuk Transformer: (250, 20)
        mfcc = mfcc.T
        return mfcc
        
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None

def load_model():
    """Load model dan normalization"""
    global model, norm_mean, norm_std
    
    try:
        print("=== LOADING MODEL ===")
        
        # Check files
        model_file = 'chicken_transformer_model.h5'
        norm_file = 'chicken_transformer_model_norm.npz'
        
        if not os.path.exists(model_file):
            print(f"❌ Model file not found: {model_file}")
            print(f"Files in directory: {os.listdir('.')}")
            return False
            
        if not os.path.exists(norm_file):
            print(f"❌ Norm file not found: {norm_file}")
            print(f"Files in directory: {os.listdir('.')}")
            return False
        
        print(f"✅ Files found:")
        print(f"   Model: {model_file} ({os.path.getsize(model_file)/1024/1024:.1f}MB)")
        print(f"   Norm: {norm_file}")
        
        print("Loading Transformer model...")
        model = tf.keras.models.load_model(model_file)
        print(f"✅ Model loaded! Input shape: {model.input_shape}")
        
        print("Loading normalization...")
        norm_data = np.load(norm_file)
        norm_mean = norm_data['mean']
        norm_std = norm_data['std']
        print(f"✅ Normalization loaded! Mean: {norm_mean:.6f}, Std: {norm_std:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def home():
    return jsonify({
        'success': True,
        'message': 'Chicken Transformer API Running',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save temporary file
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # Extract MFCC
        mfcc_features = extract_mfcc_features(temp_path)
        
        if mfcc_features is None:
            os.remove(temp_path)
            os.rmdir(temp_dir)
            return jsonify({'success': False, 'error': 'MFCC extraction failed'}), 400
        
        # Preprocess
        mfcc_batch = np.expand_dims(mfcc_features, axis=0)
        mfcc_normalized = (mfcc_batch - norm_mean) / norm_std
        mfcc_normalized = mfcc_normalized.astype('float32')
        
        # Predict
        predictions = model.predict(mfcc_normalized, verbose=0)
        
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        predicted_label = categories[predicted_class]
        
        # Cleanup
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        # Return result - FORMAT SESUAI ANDROID
        return jsonify({
            'success': True,
            'filename': filename,
            'prediction': {
                'predicted_class': predicted_label,
                'confidence': confidence,
                'all_predictions': {
                    categories[i]: float(predictions[0][i]) for i in range(len(categories))
                }
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Load model saat startup (untuk gunicorn)
print("🐔 Starting Chicken Transformer API...")
if not load_model():
    print("❌ Failed to load model")
    exit(1)
else:
    print("✅ Model loaded successfully!")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
