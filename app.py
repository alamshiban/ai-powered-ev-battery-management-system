import os
import sys
import warnings
import json
import logging
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
sys.modules['numpy._core.numeric'] = sys.modules['numpy.core.numeric']

import numpy.random._pickle
_original_ctor = numpy.random._pickle.__bit_generator_ctor

def _patched_ctor(bit_generator_name):
    if hasattr(bit_generator_name, '__name__'):
        return _original_ctor(bit_generator_name.__name__)
    if "PCG64" in str(bit_generator_name):
        return _original_ctor("PCG64")
    return _original_ctor(bit_generator_name)

numpy.random._pickle.__bit_generator_ctor = _patched_ctor

import pandas as pd
import pickle
import h5py
import gymnasium as gym
import torch
from torch.optim import Optimizer
from flask import Flask, request, jsonify
from flask_cors import CORS
from stable_baselines3 import DQN

app = Flask(__name__)
CORS(app) 

MODEL_DIR = 'models'
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

battery_state = {
    'soh': 100.0, 'soc': 80.0, 'temperature': 35.0, 'voltage': 3.7,
    'current': 1.0, 'cycle_count': 0, 'last_action': 'Normal',
    'cumulative_reward': 0.0, 'efficiency': 85.0
}

print("="*80)
print("STARTING LOCAL PRODUCTION SERVER...")
print("="*80)

def patch_keras_h5(filepath):
    try:
        with h5py.File(filepath, 'r+') as f:
            if 'model_config' in f.attrs:
                raw_config = f.attrs['model_config']
                config_str = raw_config.decode('utf-8') if isinstance(raw_config, bytes) else raw_config
                config = json.loads(config_str)
                modified = False
                if 'config' in config and 'layers' in config['config']:
                    for layer in config['config']['layers']:
                        layer_cfg = layer.get('config', {})
                        bad_keys = ['optional', 'quantization_config', 'kernel_regularizer', 
                                    'bias_regularizer', 'activity_regularizer']
                        for bad_key in bad_keys:
                            if bad_key in layer_cfg:
                                del layer_cfg[bad_key]
                                modified = True
                        if layer.get('class_name') == 'InputLayer' and 'batch_shape' in layer_cfg:
                            layer_cfg['batch_input_shape'] = layer_cfg.pop('batch_shape')
                            modified = True
                        if 'dtype' in layer_cfg and isinstance(layer_cfg['dtype'], dict):
                            dtype_dict = layer_cfg['dtype']
                            if dtype_dict.get('class_name') == 'DTypePolicy':
                                layer_cfg['dtype'] = dtype_dict.get('config', {}).get('name', 'float32')
                                modified = True
                if modified:
                    new_config_str = json.dumps(config)
                    f.attrs['model_config'] = new_config_str.encode('utf-8') if isinstance(raw_config, bytes) else new_config_str
    except Exception as e:
        pass

models = {}

try:
    for model_file in ['cnn_model.h5', 'lstm_model.h5', 'hybrid_model.h5']:
        patch_keras_h5(os.path.join(MODEL_DIR, model_file))

    print("Loading Neural Networks...")
    models['cnn'] = keras.models.load_model(os.path.join(MODEL_DIR, 'cnn_model.h5'), compile=False)
    models['lstm'] = keras.models.load_model(os.path.join(MODEL_DIR, 'lstm_model.h5'), compile=False)
    models['hybrid'] = keras.models.load_model(os.path.join(MODEL_DIR, 'hybrid_model.h5'), compile=False)
    
    print("Loading Reinforcement Learning Agents...")
    with open(os.path.join(MODEL_DIR, 'q_table_baseline.pkl'), 'rb') as f:
        models['q_table'] = pickle.load(f)
        
    custom_objects = {
        "learning_rate": 0.0, "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0, "_np_random": None,
        "observation_space": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        "action_space": gym.spaces.Discrete(3)
    }
    
    _original_load_state_dict = Optimizer.load_state_dict
    Optimizer.load_state_dict = lambda self, state_dict: None
    models['dqn'] = DQN.load(os.path.join(MODEL_DIR, 'optimized_battery_dqn'), custom_objects=custom_objects)
    Optimizer.load_state_dict = _original_load_state_dict
    
    print("✅ All Models Loaded Successfully on Local Machine!")
except Exception as e:
    print(f"❌ ERROR LOADING MODELS: {str(e)}")

def discretize_state(obs_array):
    return tuple(np.round(obs_array * 10))

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(battery_state)

@app.route('/api/predict', methods=['POST'])
def predict_soh():
    try:
        file = request.files['file']
        model_choice = request.form.get('model', 'hybrid').lower()
        
        filepath = os.path.join(UPLOAD_DIR, 'temp_telemetry.csv')
        file.save(filepath)
        df = pd.read_csv(filepath)
        latest_row = df.iloc[-1]
        
        actual_soh = None
        feature_data = None
        
        target_cols = ['SOH_target', 'SOH', 'soh']
        for col in target_cols:
            if col in df.columns:
                actual_val = float(latest_row[col])
                actual_soh = actual_val * 100 if actual_val <= 1.5 else actual_val
                feature_data = latest_row.drop(col).values
                break
                
        if feature_data is None:
            feature_data = latest_row.values

        if len(feature_data) == 60:
            # FIX 1: The frontend data is ALREADY SCALED. Do not run scaler.transform()!
            # FIX 2: Restore the correct reshape order based on the training flatten() method.
            input_tensor = feature_data.reshape(1, 10, 6).astype(np.float32)
        else:
            return jsonify({'error': f'Dataset shape mismatch. Expected 60 features, got {len(feature_data)}'}), 400
        
        if model_choice == 'cnn':
            pred = models['cnn'].predict(input_tensor, verbose=0)
        elif model_choice == 'lstm':
            pred = models['lstm'].predict(input_tensor, verbose=0)
        else: 
            pred = models['hybrid'].predict(input_tensor, verbose=0)
            
        raw_pred = float(pred[0][0])
        predicted_soh = raw_pred * 100 if raw_pred <= 1.5 else raw_pred
        predicted_soh = max(0.0, min(100.0, predicted_soh))
        
        battery_state['soh'] = predicted_soh
        
        return jsonify({
            'success': True,
            'predicted_soh': round(predicted_soh, 2),
            'actual_soh': round(actual_soh, 2) if actual_soh else None,
            'status': 'Healthy' if predicted_soh > 80 else 'Degraded'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/optimize', methods=['POST'])
def optimize_discharge():
    try:
        data = request.json or {}
        rl_model_choice = data.get('algorithm', 'dqn').lower()
        dynamic_load = np.sin(battery_state['cycle_count'] / 5.0) 
        
        obs = np.array([
            battery_state['voltage'] + (dynamic_load * 0.2), 
            battery_state['current'] + (dynamic_load * 0.1), 
            battery_state['temperature'], 
            battery_state['soc'] / 100.0, 
            battery_state['efficiency'] / 100.0, 
            battery_state['soh'] / 100.0
        ], dtype=np.float32).reshape(1, -1)

        if random.random() < 0.15:
            action = random.choice([0, 1, 2])
        else:
            if rl_model_choice == 'q_learning':
                state_tuple = discretize_state(obs[0])
                action = int(np.argmax(models['q_table'][state_tuple])) if state_tuple in models['q_table'] else 1 
            else: 
                try:
                    action, _ = models['dqn'].predict(obs, deterministic=True)
                    action = int(action[0]) if isinstance(action, np.ndarray) else int(action)
                except:
                    action = 1

        action_map = {0: "Aggressive", 1: "Moderate", 2: "Conservative"}
        chosen_action = action_map.get(action, "Moderate")
        
        if action == 0:
            efficiency, temp_change, soc_drain, reward = 95.0, 1.5, 1.5, 5.0
        elif action == 1:
            efficiency, temp_change, soc_drain, reward = 85.0, 0.2, 0.8, 2.0
        else:
            efficiency, temp_change, soc_drain, reward = 75.0, -0.5, 0.4, 1.0 

        ambient_temp = 35.0
        temp_recovery = (ambient_temp - battery_state['temperature']) * 0.2
        
        battery_state['last_action'] = chosen_action
        battery_state['efficiency'] = efficiency
        battery_state['cumulative_reward'] += reward
        
        new_temp = battery_state['temperature'] + temp_change + temp_recovery
        battery_state['temperature'] = max(25.0, min(65.0, new_temp))
        
        battery_state['soc'] = max(0.0, battery_state['soc'] - soc_drain)
        battery_state['cycle_count'] += 1

        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_state():
    global battery_state
    battery_state = {
        'soh': 100.0, 'soc': 80.0, 'temperature': 35.0, 'voltage': 3.7,
        'current': 1.0, 'cycle_count': 0, 'last_action': 'Normal',
        'cumulative_reward': 0.0, 'efficiency': 85.0
    }
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, use_reloader=False)