from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
from werkzeug.utils import secure_filename
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

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

def preprocess_audio_for_consistency(audio_path):
    """
    Enhanced audio preprocessing untuk consistency dengan training data
    """
    try:
        print(f"Processing audio: {audio_path}")
        
        # Load audio dengan librosa (sama dengan training)
        audio, original_sr = librosa.load(audio_path, sr=None)
        print(f"Original: SR={original_sr}, Duration={len(audio)/original_sr:.2f}s, Shape={audio.shape}")
        
        # === AUDIO QUALITY IMPROVEMENTS ===
        
        # 1. Convert ke mono jika stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
            print("✅ Converted stereo to mono")
        
        # 2. Resample ke sample rate standar (gunakan 22050 Hz yang umum untuk speech)
        target_sr = 22050
        if original_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
            print(f"✅ Resampled {original_sr}Hz -> {target_sr}Hz")
        else:
            target_sr = original_sr
        
        # 3. Normalisasi amplitude (prevent clipping)
        audio = audio / np.max(np.abs(audio))
        print("✅ Normalized amplitude")
        
        # 4. Noise reduction sederhana (trim silence)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        print(f"✅ Trimmed silence, new duration: {len(audio)/target_sr:.2f}s")
        
        # 5. Filter noise dengan spectral gating sederhana
        # Hitung noise floor
        noise_floor = np.percentile(np.abs(audio), 10)
        # Soft gating (jangan terlalu aggressive)
        audio = np.where(np.abs(audio) < noise_floor * 0.5, audio * 0.1, audio)
        print("✅ Applied noise reduction")
        
        # 6. Ensure minimum duration (jika terlalu pendek, repeat)
        min_duration = 1.0  # seconds
        min_samples = int(min_duration * target_sr)
        if len(audio) < min_samples:
            # Repeat audio until minimum duration
            repeats = int(np.ceil(min_samples / len(audio)))
            audio = np.tile(audio, repeats)[:min_samples]
            print(f"✅ Extended short audio to {min_duration}s")
        
        # === MFCC EXTRACTION (sama dengan training) ===
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=target_sr,
            n_mfcc=N_MFCC,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            window='hann',  # Explicitly set window
            center=True,    # Center frames
            power=2.0       # Use power spectrum
        )
        
        print(f"MFCC shape before padding: {mfcc.shape}")
        
        # Padding/truncate (sama dengan training)
        if mfcc.shape[1] < MAX_LENGTH:
            pad_width = MAX_LENGTH - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        elif mfcc.shape[1] > MAX_LENGTH:
            mfcc = mfcc[:, :MAX_LENGTH]
        
        # Transpose untuk Transformer: (250, 20)
        mfcc = mfcc.T
        print(f"Final MFCC shape: {mfcc.shape}")
        
        return mfcc
        
    except Exception as e:
        print(f"Error in audio preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_mfcc_features(audio_path):
    """Extract MFCC features dengan enhanced preprocessing"""
    return preprocess_audio_for_consistency(audio_path)

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
        'message': 'Chicken Transformer API Running - Enhanced Audio Processing',
        'model_loaded': model is not None,
        'audio_preprocessing': {
            'noise_reduction': True,
            'amplitude_normalization': True,
            'sample_rate_standardization': True,
            'silence_trimming': True
        }
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
        
        print(f"\n=== PROCESSING REQUEST ===")
        print(f"File: {file.filename}")
        
        # Save temporary file
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        print(f"Saved to: {temp_path}")
        print(f"File size: {os.path.getsize(temp_path)} bytes")
        
        # Extract MFCC dengan enhanced preprocessing
        mfcc_features = extract_mfcc_features(temp_path)
        
        if mfcc_features is None:
            os.remove(temp_path)
            os.rmdir(temp_dir)
            return jsonify({'success': False, 'error': 'MFCC extraction failed'}), 400
        
        # Preprocess untuk model (sama dengan training)
        mfcc_batch = np.expand_dims(mfcc_features, axis=0)
        mfcc_normalized = (mfcc_batch - norm_mean) / norm_std
        mfcc_normalized = mfcc_normalized.astype('float32')
        
        print(f"Input to model: {mfcc_normalized.shape}")
        
        # Predict
        print("Running prediction...")
        predictions = model.predict(mfcc_normalized, verbose=0)
        
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        predicted_label = categories[predicted_class]
        
        print(f"Prediction: {predicted_label} (confidence: {confidence:.4f})")
        
        # Show all predictions untuk debugging
        print("All predictions:")
        for i, cat in enumerate(categories):
            print(f"  {cat}: {predictions[0][i]:.4f}")
        
        # Cleanup
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        # Return result dengan info tambahan
        return jsonify({
            'success': True,
            'filename': filename,
            'prediction': {
                'predicted_class': predicted_label,
                'confidence': confidence,
                'all_predictions': {
                    categories[i]: float(predictions[0][i]) for i in range(len(categories))
                }
            },
            'processing_info': {
                'mfcc_shape': list(mfcc_features.shape),
                'preprocessing_applied': [
                    'mono_conversion',
                    'sample_rate_standardization', 
                    'amplitude_normalization',
                    'noise_reduction',
                    'silence_trimming'
                ]
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check dengan info model"""
    return jsonify({
        'success': True,
        'model_loaded': model is not None,
        'tensorflow_version': tf.__version__,
        'categories': categories,
        'mfcc_config': {
            'n_mfcc': N_MFCC,
            'hop_length': HOP_LENGTH,
            'n_fft': N_FFT,
            'max_length': MAX_LENGTH
        }
    })

# Load model saat startup
print("🐔 Starting Enhanced Chicken Transformer API...")
if not load_model():
    print("❌ Failed to load model")
    exit(1)
else:
    print("✅ Model loaded successfully!")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
