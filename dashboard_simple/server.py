#!/usr/bin/env python3
"""Simple Dashboard Server for Enlitens AI Processing"""

from flask import Flask, jsonify, send_file
from flask_cors import CORS
import psutil
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

app = Flask(__name__, static_folder='.')
CORS(app)

PROJECT_ROOT = Path("/home/antons-gs/enlitens-ai")
LOG_FILE = PROJECT_ROOT / "logs" / "enlitens_complete_processing.log"
JSON_FILE = PROJECT_ROOT / "enlitens_knowledge_base.json.temp"

def get_gpu_stats():
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=2)
        
        if result.returncode == 0:
            util, mem_used, mem_total, temp, power = result.stdout.strip().split(', ')
            return {
                'utilization': int(util),
                'memory_used': float(mem_used) / 1024,
                'memory_total': float(mem_total) / 1024,
                'temperature': int(float(temp)),
                'power': int(float(power))
            }
    except:
        pass
    
    return {'utilization': 0, 'memory_used': 0, 'memory_total': 24, 'temperature': 0, 'power': 0}

def get_cpu_stats():
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        load_avg = os.getloadavg()
        
        temp = 0
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                temp = int(temps['coretemp'][0].current)
        except:
            pass
        
        return {
            'utilization': int(cpu_percent),
            'memory_used': round(mem.used / (1024**3), 1),
            'memory_total': round(mem.total / (1024**3), 1),
            'temperature': temp,
            'load_avg': f"{load_avg[0]:.2f}"
        }
    except:
        return {'utilization': 0, 'memory_used': 0, 'memory_total': 64, 'temperature': 0, 'load_avg': "0.00"}

def get_processing_stats():
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()
                
                start_index = 0
                for i in range(len(lines) - 1, -1, -1):
                    if '🚀 Starting MULTI-AGENT' in lines[i]:
                        start_index = i
                        break
                
                current_run_lines = lines[start_index:]
                completed = sum(1 for line in current_run_lines if '✅ Document' in line and 'processed successfully' in line)
                
                current_file = 'None'
                stage = 'Idle'
                
                for line in reversed(current_run_lines[-100:]):
                    if '📖 Processing file' in line:
                        parts = line.split('Processing file')
                        if len(parts) > 1:
                            current_file = parts[1].split(':')[1].strip() if ':' in parts[1] else 'Unknown'
                        break
                
                for line in reversed(current_run_lines[-50:]):
                    if 'Agent' in line and 'starting processing' in line:
                        stage = line.split('Agent')[1].split('starting')[0].strip()
                        break
                
                return {
                    'completed': completed,
                    'total': 345,
                    'current_file': current_file,
                    'stage': stage
                }
    except:
        pass
    
    return {'completed': 0, 'total': 345, 'current_file': 'None', 'stage': 'Idle'}

def get_json_stats():
    try:
        if JSON_FILE.exists():
            stat = JSON_FILE.stat()
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)
                return {
                    'size': stat.st_size,
                    'documents': len(data.get('documents', [])),
                    'updated': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
    except:
        pass
    
    return {'size': 0, 'documents': 0, 'updated': datetime.now().isoformat()}

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/api/metrics')
def metrics():
    return jsonify({
        'gpu': get_gpu_stats(),
        'cpu': get_cpu_stats(),
        'processing': get_processing_stats(),
        'json': get_json_stats()
    })

@app.route('/api/logs')
def logs():
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()
                start_index = 0
                for i in range(len(lines) - 1, -1, -1):
                    if '🚀 Starting MULTI-AGENT' in lines[i]:
                        start_index = i
                        break
                current_run_lines = lines[start_index:]
                recent_logs = current_run_lines[-200:]
                return jsonify({'logs': [line.strip() for line in recent_logs]})
    except:
        pass
    return jsonify({'logs': ['No logs available']})

@app.route('/api/download')
def download():
    if JSON_FILE.exists():
        return send_file(JSON_FILE, as_attachment=True, download_name='enlitens_knowledge_base.json')
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    print("🚀 Starting Enlitens Dashboard Server...")
    print("📊 Dashboard: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
