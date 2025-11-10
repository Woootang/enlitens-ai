#!/usr/bin/env python3
"""
Enhanced Dashboard Server for Enlitens AI Processing
Provides comprehensive system monitoring and processing analytics
"""

from flask import Flask, jsonify, send_file
from flask_cors import CORS
import psutil
import json
import os
import subprocess
import re
import time
from datetime import datetime
from pathlib import Path

app = Flask(__name__, static_folder='.')
CORS(app)

PROJECT_ROOT = Path("/home/antons-gs/enlitens-ai")
LOG_FILE = PROJECT_ROOT / "logs" / "enlitens_complete_processing.log"
JSON_FILE = PROJECT_ROOT / "enlitens_knowledge_base" / "enlitens_knowledge_base.json.temp"
TOTAL_DOCUMENTS = 344

# Cache for expensive log parsing
_CACHE = {"timestamp": 0, "data": None}

def get_system_power():
    """Estimate total system power draw"""
    try:
        # Get GPU power
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=2)
        
        gpu_power = 0
        if result.returncode == 0:
            gpu_power = float(result.stdout.strip())
        
        # Estimate CPU power (rough approximation based on usage)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_tdp = 125  # Typical desktop CPU TDP
        cpu_power = (cpu_percent / 100) * cpu_tdp
        
        # Add estimated motherboard/RAM/storage power (~50-100W)
        system_overhead = 75
        
        total_power = gpu_power + cpu_power + system_overhead
        
        return {
            'total': round(total_power),
            'gpu': round(gpu_power),
            'cpu': round(cpu_power),
            'other': system_overhead,
            'psu_capacity': 1000,
            'psu_usage_percent': round((total_power / 1000) * 100, 1)
        }
    except:
        return {'total': 0, 'gpu': 0, 'cpu': 0, 'other': 0, 'psu_capacity': 1000, 'psu_usage_percent': 0}

def get_gpu_stats():
    """Get comprehensive GPU metrics"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=2)
        
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            util, mem_used, mem_total, temp, power = parts[:5]
            fan = parts[5] if len(parts) > 5 else '0'
            
            return {
                'utilization': int(util),
                'memory_used': round(float(mem_used) / 1024, 2),
                'memory_total': round(float(mem_total) / 1024, 2),
                'temperature': int(float(temp)),
                'power': int(float(power)),
                'fan_speed': int(fan) if fan != '[N/A]' else 0
            }
    except:
        pass
    
    return {
        'utilization': 0,
        'memory_used': 0,
        'memory_total': 24,
        'temperature': 0,
        'power': 0,
        'fan_speed': 0
    }

def get_cpu_stats():
    """Get comprehensive CPU metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        load_avg = os.getloadavg()
        cpu_freq = psutil.cpu_freq()
        
        # Get CPU temperature
        temp = 0
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                temp = int(temps['coretemp'][0].current)
            elif 'k10temp' in temps:
                temp = int(temps['k10temp'][0].current)
        except:
            pass
        
        return {
            'utilization': int(cpu_percent),
            'memory_used': round(mem.used / (1024**3), 1),
            'memory_total': round(mem.total / (1024**3), 1),
            'memory_percent': int(mem.percent),
            'temperature': temp,
            'load_avg': f"{load_avg[0]:.2f}",
            'frequency': round(cpu_freq.current) if cpu_freq else 0,
            'cores': psutil.cpu_count()
        }
    except:
        return {
            'utilization': 0,
            'memory_used': 0,
            'memory_total': 64,
            'memory_percent': 0,
            'temperature': 0,
            'load_avg': "0.00",
            'frequency': 0,
            'cores': 0
        }

def parse_logs():
    """Parse logs and extract comprehensive processing analytics"""
    now = time.time()
    if _CACHE["data"] and (now - _CACHE["timestamp"]) < 1.0:
        return _CACHE["data"]
    
    if not LOG_FILE.exists():
        return {"processing": {}, "agents": [], "alerts": {}}
    
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
    
    # Find current run start
    start_idx = 0
    for i in range(len(lines) - 1, -1, -1):
        if '🚀 Starting MULTI-AGENT' in lines[i]:
            start_idx = i
            break
    
    current_lines = lines[start_idx:]
    
    # Initialize data structures
    processing = {
        'completed': 0,
        'total': TOTAL_DOCUMENTS,
        'current_file': 'None',
        'stage': 'Idle',
        'start_time': None,
        'latest_doc': {}
    }
    
    # Agent metadata with proper names and descriptions
    agent_metadata = {
        'supervisor': {
            'name': 'Supervisor',
            'emoji': '🎯',
            'description': 'Orchestrates the entire multi-agent pipeline',
            'role': 'Coordinator'
        },
        'context_curator': {
            'name': 'Context Curator',
            'emoji': '🧬',
            'description': 'Curates relevant personas and contextual information',
            'role': 'Context Builder'
        },
        'pdf_extraction': {
            'name': 'PDF Extractor',
            'emoji': '📄',
            'description': 'Extracts text and structure from research papers',
            'role': 'Document Parser'
        },
        'science_extraction': {
            'name': 'Science Extractor',
            'emoji': '🔬',
            'description': 'Identifies research methods, findings, and statistics',
            'role': 'Research Analyzer'
        },
        'clinical_synthesis': {
            'name': 'Clinical Synthesizer',
            'emoji': '⚕️',
            'description': 'Synthesizes clinical implications and applications',
            'role': 'Clinical Translator'
        },
        'educational_content': {
            'name': 'Educational Content',
            'emoji': '📚',
            'description': 'Generates educational materials and summaries',
            'role': 'Content Creator'
        },
        'rebellion_framework': {
            'name': 'Rebellion Framework',
            'emoji': '🔥',
            'description': 'Applies rebellious narrative and empowerment framing',
            'role': 'Narrative Designer'
        },
        'founder_voice': {
            'name': 'Founder Voice',
            'emoji': '🗣️',
            'description': 'Ensures content matches founder\'s authentic voice',
            'role': 'Voice Authenticator'
        },
        'marketing_seo': {
            'name': 'Marketing & SEO',
            'emoji': '📈',
            'description': 'Optimizes content for search and engagement',
            'role': 'Growth Optimizer'
        },
        'validation': {
            'name': 'Validator',
            'emoji': '✅',
            'description': 'Validates output quality and completeness',
            'role': 'Quality Assurance'
        },
        'profile_matcher': {
            'name': 'Profile Matcher',
            'emoji': '🎭',
            'description': 'Matches content to relevant client personas',
            'role': 'Persona Selector'
        },
        'voice_guide_generator': {
            'name': 'Voice Guide',
            'emoji': '🎤',
            'description': 'Generates voice and tone guidelines',
            'role': 'Style Guide'
        },
        'health_report_synthesizer': {
            'name': 'Health Synthesizer',
            'emoji': '🏥',
            'description': 'Synthesizes regional health context',
            'role': 'Health Analyst'
        }
    }
    
    agents = {}
    alerts = {'count': 0, 'last_error': None}
    context_curator = {'personas': 0, 'health': 0, 'voice': 0, 'total': 0}
    
    # Parse logs
    for line in current_lines:
        # Count completed documents
        if '✅ Document' in line and 'processed successfully' in line:
            processing['completed'] += 1
            
            # Extract document info
            match = re.search(r'Document (.+?) processed successfully in ([0-9.]+)s', line)
            if match:
                processing['latest_doc'] = {
                    'id': match.group(1),
                    'duration': round(float(match.group(2)) / 60, 2)
                }
        
        # Extract quality/confidence
        if 'Quality' in line and 'Confidence' in line:
            match = re.search(r'Quality ([0-9.]+) Confidence ([0-9.]+)', line)
            if match:
                processing['latest_doc']['quality'] = float(match.group(1))
                processing['latest_doc']['confidence'] = float(match.group(2))
        
        # Current file
        if '📖 Processing file' in line:
            match = re.search(r'Processing file \d+/\d+:\s*(.+)', line)
            if match:
                processing['current_file'] = match.group(1).strip()
        
        # Current stage
        if 'Agent' in line and 'starting processing' in line:
            match = re.search(r'Agent ([A-Za-z0-9_]+)', line)
            if match:
                processing['stage'] = match.group(1)
        
        # Agent status - more comprehensive parsing
        if 'Agent' in line or any(agent_key in line.lower() for agent_key in agent_metadata.keys()):
            # Try to match agent name from logs
            for agent_key, metadata in agent_metadata.items():
                if agent_key.lower() in line.lower() or metadata['name'].lower() in line.lower():
                    if agent_key not in agents:
                        agents[agent_key] = {
                            'id': agent_key,
                            'name': metadata['name'],
                            'emoji': metadata['emoji'],
                            'description': metadata['description'],
                            'role': metadata['role'],
                            'status': 'idle',
                            'last_action': None,
                            'processing_time': None
                        }
                    
                    # Update status based on log content
                    if 'starting' in line.lower() or 'processing' in line.lower():
                        agents[agent_key]['status'] = 'running'
                        agents[agent_key]['last_action'] = 'Processing...'
                    elif 'completed' in line.lower() or 'successfully' in line.lower():
                        agents[agent_key]['status'] = 'completed'
                        agents[agent_key]['last_action'] = 'Completed'
                        # Try to extract processing time
                        time_match = re.search(r'in ([0-9.]+)s', line)
                        if time_match:
                            agents[agent_key]['processing_time'] = f"{float(time_match.group(1)):.2f}s"
                    elif 'failed' in line.lower() or 'error' in line.lower():
                        agents[agent_key]['status'] = 'error'
                        agents[agent_key]['last_action'] = 'Error occurred'
                    break
        
        # Context curator tokens
        if 'tokens' in line.lower():
            if 'personas' in line.lower():
                match = re.search(r'(\d+)\s*tokens', line)
                if match:
                    context_curator['personas'] = int(match.group(1))
            elif 'health' in line.lower() or 'brief' in line.lower():
                match = re.search(r'(\d+)\s*tokens', line)
                if match:
                    context_curator['health'] = int(match.group(1))
            elif 'voice' in line.lower() or 'guide' in line.lower():
                match = re.search(r'(\d+)\s*tokens', line)
                if match:
                    context_curator['voice'] = int(match.group(1))
        
        # Errors
        if 'ERROR' in line or '❌' in line:
            alerts['count'] += 1
            alerts['last_error'] = line.strip()[-200:]  # Last 200 chars
    
    # Calculate context curator total
    context_curator['total'] = context_curator['personas'] + context_curator['health'] + context_curator['voice']
    
    # Get start time
    for line in current_lines[:50]:
        if '🚀 Starting MULTI-AGENT' in line:
            try:
                timestamp_str = line[:19]
                processing['start_time'] = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').isoformat()
            except:
                pass
            break
    
    result = {
        'processing': processing,
        'agents': list(agents.values()),
        'alerts': alerts,
        'context_curator': context_curator
    }
    
    _CACHE["data"] = result
    _CACHE["timestamp"] = now
    return result

def get_json_stats():
    """Get knowledge base JSON statistics"""
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
    """Get all system metrics"""
    log_data = parse_logs()
    
    return jsonify({
        'power': get_system_power(),
        'gpu': get_gpu_stats(),
        'cpu': get_cpu_stats(),
        'processing': log_data['processing'],
        'agents': log_data['agents'],
        'alerts': log_data['alerts'],
        'context_curator': log_data['context_curator'],
        'json': get_json_stats()
    })

@app.route('/api/chain_of_thought')
def chain_of_thought():
    """Get the AI's chain-of-thought reasoning for current document"""
    try:
        if not LOG_FILE.exists():
            return jsonify({"reasoning_steps": [], "current_document": None})
        
        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Find the most recent document being processed
        current_doc = None
        for line in reversed(lines[-500:]):  # Check last 500 lines
            if '📖 Processing file' in line or '🧠 Starting multi-agent processing' in line:
                # Extract document name
                parts = line.split(':')
                if len(parts) > 2:
                    current_doc = parts[-1].strip()
                break
        
        # Extract reasoning steps (last 100 lines)
        reasoning_steps = []
        reasoning_patterns = {
            '🎯': 'Planning',
            '🔍': 'Analyzing',
            '✅': 'Completed',
            '🧠': 'Thinking',
            '📊': 'Evaluating',
            '🏥': 'Synthesizing',
            '📝': 'Extracting',
            '⚙️': 'Processing',
            '🔄': 'Loading',
            'INFO - 🎯': 'Strategy',
            'INFO - Starting': 'Initiating',
            'agent': 'Agent Action'
        }
        
        for line in lines[-200:]:  # Last 200 lines
            line_lower = line.lower()
            # Skip debug/system lines
            if any(skip in line_lower for skip in ['gpu', 'memory', 'unloading', 'cache cleared']):
                continue
                
            # Find reasoning indicators
            for pattern, label in reasoning_patterns.items():
                if pattern.lower() in line_lower:
                    # Extract timestamp and message
                    parts = line.split(' - ')
                    if len(parts) >= 3:
                        timestamp = parts[0].strip()
                        message = ' - '.join(parts[2:]).strip()
                        
                        reasoning_steps.append({
                            'timestamp': timestamp,
                            'type': label,
                            'message': message[:200],  # Limit length
                            'icon': pattern if pattern in reasoning_patterns else '🤔'
                        })
                    break
        
        # Keep only last 20 steps
        reasoning_steps = reasoning_steps[-20:]
        
        return jsonify({
            "reasoning_steps": reasoning_steps,
            "current_document": current_doc,
            "step_count": len(reasoning_steps)
        })
    except Exception as e:
        return jsonify({"reasoning_steps": [], "error": str(e)})

@app.route('/api/logs')
def logs():
    """Get recent logs from current run"""
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()
                
                start_idx = 0
                for i in range(len(lines) - 1, -1, -1):
                    if '🚀 Starting MULTI-AGENT' in lines[i]:
                        start_idx = i
                        break
                
                current_lines = lines[start_idx:]
                recent = current_lines[-200:]
                return jsonify({'logs': [line.strip() for line in recent]})
    except:
        pass
    
    return jsonify({'logs': ['No logs available']})

@app.route('/api/download')
def download():
    """Download knowledge base JSON"""
    if JSON_FILE.exists():
        return send_file(JSON_FILE, as_attachment=True, download_name='enlitens_knowledge_base.json')
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    print("🚀 Starting Enhanced Enlitens Dashboard...")
    print("📊 Dashboard: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
