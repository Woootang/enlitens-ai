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

# Agent metadata shared across endpoints
AGENT_METADATA = {
    'supervisor': {
        'name': 'Supervisor',
        'emoji': '🎯',
        'description': 'Orchestrates the entire multi-agent pipeline',
        'role': 'Coordinator',
        'keywords': ['supervisor', 'overall supervisor']
    },
    'context_curator': {
        'name': 'Context Curator',
        'emoji': '🧬',
        'description': 'Curates relevant personas and contextual information',
        'role': 'Context Builder',
        'keywords': ['context_curator', 'context curator']
    },
    'pdf_extraction': {
        'name': 'PDF Extractor',
        'emoji': '📄',
        'description': 'Extracts text and structure from research papers',
        'role': 'Document Parser',
        'keywords': ['pdf_extractor', 'pdf extraction']
    },
    'science_extraction': {
        'name': 'Science Extractor',
        'emoji': '🔬',
        'description': 'Identifies research methods, findings, and statistics',
        'role': 'Research Analyzer',
        'keywords': ['science_extraction', 'science extractor']
    },
    'clinical_synthesis': {
        'name': 'Clinical Synthesizer',
        'emoji': '⚕️',
        'description': 'Synthesizes clinical implications and applications',
        'role': 'Clinical Translator',
        'keywords': ['clinical_synthesis', 'clinical synthesizer']
    },
    'educational_content': {
        'name': 'Educational Content',
        'emoji': '📚',
        'description': 'Generates educational materials and summaries',
        'role': 'Content Creator',
        'keywords': ['educational_content']
    },
    'rebellion_framework': {
        'name': 'Rebellion Framework',
        'emoji': '🔥',
        'description': 'Applies rebellious narrative and empowerment framing',
        'role': 'Narrative Designer',
        'keywords': ['rebellion_framework']
    },
    'founder_voice': {
        'name': 'Founder Voice',
        'emoji': '🗣️',
        'description': 'Ensures content matches founder\'s authentic voice',
        'role': 'Voice Authenticator',
        'keywords': ['founder_voice', 'voice guide']
    },
    'marketing_seo': {
        'name': 'Marketing & SEO',
        'emoji': '📈',
        'description': 'Optimizes content for search and engagement',
        'role': 'Growth Optimizer',
        'keywords': ['marketing', 'seo']
    },
    'validation': {
        'name': 'Validator',
        'emoji': '✅',
        'description': 'Validates output quality and completeness',
        'role': 'Quality Assurance',
        'keywords': ['validation']
    },
    'profile_matcher': {
        'name': 'Profile Matcher',
        'emoji': '🎭',
        'description': 'Matches content to relevant client personas',
        'role': 'Persona Selector',
        'keywords': ['profile_matcher', 'persona matcher']
    },
    'voice_guide_generator': {
        'name': 'Voice Guide Generator',
        'emoji': '🎤',
        'description': 'Generates voice and tone guidelines',
        'role': 'Style Guide',
        'keywords': ['voice_guide_generator', 'voice guide generator']
    },
    'health_report_synthesizer': {
        'name': 'Health Report Synthesizer',
        'emoji': '🏥',
        'description': 'Synthesizes regional health context',
        'role': 'Health Analyst',
        'keywords': ['health_report_synthesizer', 'health report synthesizer']
    }
}

AGENT_KEYWORDS = []
for agent_key, meta in AGENT_METADATA.items():
    AGENT_KEYWORDS.append((agent_key, meta['name'].lower()))
    for keyword in meta.get('keywords', []):
        AGENT_KEYWORDS.append((agent_key, keyword.lower()))

AGENT_KEYWORD_STRINGS = [keyword for _, keyword in AGENT_KEYWORDS]


def identify_agent(line: str):
    """Return the agent key that matches a log line, or None."""
    line_lower = line.lower()
    for agent_key, keyword in AGENT_KEYWORDS:
        if keyword in line_lower:
            return agent_key
    return None

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
    
    agents = {}
    alerts = {'count': 0, 'last_error': None, 'entries': []}
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
        if 'agent' in line.lower() or any(keyword in line.lower() for keyword in AGENT_KEYWORD_STRINGS):
            # Try to match agent name from logs
            for agent_key, metadata in AGENT_METADATA.items():
                if agent_key in line.lower() or metadata['name'].lower() in line.lower():
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
        if any(tag in line for tag in ['ERROR', '❌', 'CRITICAL', 'WARNING']):
            alerts['count'] += 1
            parts = line.split(' - ', 2)
            timestamp = parts[0].strip() if parts else ''
            level = 'INFO'
            if 'CRITICAL' in line:
                level = 'CRITICAL'
            elif 'ERROR' in line or '❌' in line:
                level = 'ERROR'
            elif 'WARNING' in line:
                level = 'WARNING'
            message = parts[-1].strip() if len(parts) >= 3 else line.strip()
            alerts['entries'].append({
                'timestamp': timestamp,
                'level': level,
                'message': message,
                'raw': line.strip()
            })
    
    # Calculate context curator total
    context_curator['total'] = context_curator['personas'] + context_curator['health'] + context_curator['voice']
    alerts['entries'] = alerts['entries'][-25:]
    if alerts['entries']:
        alerts['last_error'] = alerts['entries'][-1]['message']
    
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
    """Get structured and raw chain-of-thought reasoning grouped by agent."""
    try:
        if not LOG_FILE.exists():
            return jsonify({
                "timeline": [],
                "agents": [],
                "raw_excerpt": [],
                "current_document": None,
                "step_count": 0
            })

        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Determine the most recent document being processed
        current_doc = None
        for line in reversed(lines[-600:]):
            if '📖 Processing file' in line or '🧠 Starting multi-agent processing' in line:
                parts = line.split(':')
                if len(parts) > 2:
                    current_doc = parts[-1].strip()
                break

        recent_lines = lines[-600:]
        timeline = []
        agent_buckets = {}
        raw_excerpt = []

        for raw_line in recent_lines:
            stripped = raw_line.strip()
            if not stripped:
                continue

            raw_excerpt.append(stripped)
            parts = raw_line.split(' - ', 2)
            timestamp = parts[0].strip() if parts else ''
            remainder = parts[-1].strip() if len(parts) >= 3 else stripped

            level = 'INFO'
            if 'CRITICAL' in raw_line:
                level = 'CRITICAL'
            elif 'ERROR' in raw_line or '❌' in raw_line:
                level = 'ERROR'
            elif 'WARNING' in raw_line:
                level = 'WARNING'

            agent_key = identify_agent(raw_line) or 'system'
            agent_meta = AGENT_METADATA.get(agent_key, {
                'name': 'System',
                'emoji': '🛰️',
                'description': 'Pipeline event',
                'role': 'Runtime'
            })

            is_thought = '🧠 thinking' in raw_line.lower() or 'thinking:' in remainder.lower()
            if 'THINKING:' in remainder:
                message = remainder.split('THINKING:', 1)[-1].strip()
            else:
                message = remainder

            entry = {
                'timestamp': timestamp,
                'agent_key': agent_key,
                'agent': agent_meta['name'],
                'icon': agent_meta['emoji'],
                'level': level,
                'message': message,
                'raw': stripped,
                'is_thought': is_thought
            }
            timeline.append(entry)

            bucket = agent_buckets.setdefault(agent_key, {
                'agent_key': agent_key,
                'name': agent_meta['name'],
                'icon': agent_meta['emoji'],
                'role': agent_meta.get('role', ''),
                'description': agent_meta.get('description', ''),
                'steps': []
            })
            bucket['steps'].append({
                'timestamp': timestamp,
                'message': message,
                'raw': stripped,
                'level': level,
                'is_thought': is_thought
            })

        # Limit history to keep payload manageable
        timeline = timeline[-150:]
        raw_excerpt = raw_excerpt[-250:]
        agent_cards = []
        for bucket in agent_buckets.values():
            bucket['steps'] = bucket['steps'][-12:]
            agent_cards.append(bucket)

        agent_cards.sort(key=lambda b: b['steps'][-1]['timestamp'] if b['steps'] else '', reverse=True)

        return jsonify({
            "timeline": timeline,
            "agents": agent_cards,
            "raw_excerpt": raw_excerpt,
            "current_document": current_doc,
            "step_count": len(timeline)
        })
    except Exception as e:
        return jsonify({
            "timeline": [],
            "agents": [],
            "raw_excerpt": [],
            "current_document": None,
            "error": str(e),
            "step_count": 0
        })

@app.route('/api/logs')
def logs():
    """Get recent logs from current run in CLI-friendly format"""
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
                recent = [line.rstrip('\n') for line in current_lines[-400:]]
                return jsonify({
                    'lines': recent,
                    'raw': '\n'.join(recent),
                    'logs': recent,
                    'latest': recent[-1] if recent else ''
                })
    except Exception as exc:
        return jsonify({'lines': ['Log read error: {}'.format(exc)], 'raw': str(exc), 'logs': []})

    return jsonify({'lines': ['No logs available'], 'raw': '', 'logs': []})

@app.route('/api/download')
def download():
    """Download knowledge base JSON"""
    if JSON_FILE.exists():
        return send_file(JSON_FILE, as_attachment=True, download_name='enlitens_knowledge_base.json')
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/json_preview')
def json_preview():
    """Return a preview of the most recent JSON knowledge base entries."""
    if not JSON_FILE.exists():
        return jsonify({
            'documents': [],
            'total': 0,
            'last_updated': None,
            'error': 'Knowledge base file not found.'
        })

    try:
        with open(JSON_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
            if not content:
                return jsonify({
                    'documents': [],
                    'total': 0,
                    'last_updated': datetime.fromtimestamp(JSON_FILE.stat().st_mtime).isoformat(),
                    'error': 'Knowledge base file is empty.'
                })
            data = json.loads(content)

        if isinstance(data, list):
            documents = data
        else:
            documents = data.get('documents', [])

        preview = documents[-5:]
        return jsonify({
            'documents': preview,
            'total': len(documents),
            'last_updated': datetime.fromtimestamp(JSON_FILE.stat().st_mtime).isoformat()
        })
    except json.JSONDecodeError as exc:
        return jsonify({
            'documents': [],
            'total': 0,
            'last_updated': datetime.fromtimestamp(JSON_FILE.stat().st_mtime).isoformat(),
            'error': f'JSON decode error: {exc}'
        })
    except Exception as exc:
        return jsonify({
            'documents': [],
            'total': 0,
            'last_updated': datetime.fromtimestamp(JSON_FILE.stat().st_mtime).isoformat(),
            'error': str(exc)
        })

if __name__ == '__main__':
    print("🚀 Starting Enhanced Enlitens Dashboard...")
    print("📊 Dashboard: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
