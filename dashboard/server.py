#!/usr/bin/env python3
"""
Enhanced Dashboard Server for Enlitens AI Processing
Provides comprehensive system monitoring and processing analytics
"""

from flask import Flask, jsonify, send_file, request, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
import psutil
import json
import os
import subprocess
import re
import time
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

app = Flask(__name__, static_folder='static')
CORS(app)

PROJECT_ROOT = Path("/home/antons-gs/enlitens-ai")
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.usage_tracker import get_usage_summary

# Local comparison pipeline status file
LOCAL_STATUS_FILE = PROJECT_ROOT / "logs" / "local_status.json"

# NEW PIPELINE PATHS (November 2025 rebuild)
LOG_FILE = PROJECT_ROOT / "logs" / "processing.log"
LEDGER_FILE = PROJECT_ROOT / "data" / "knowledge_base" / "enliten_knowledge_base.jsonl"
MAIN_KB_FILE = PROJECT_ROOT / "data" / "knowledge_base" / "main_kb.jsonl"

# OLD PATHS (for backward compatibility)
OLD_LOG_FILE = PROJECT_ROOT / "logs" / "enlitens_complete_processing.log"
JSON_FILE = PROJECT_ROOT / "enlitens_knowledge_base" / "enlitens_knowledge_base.json.temp"
SCIENCE_ENTRIES_FILE = PROJECT_ROOT / "data" / "knowledge_base" / "science_entries.jsonl"


def _stream_jsonl_preview(file_path: Path, max_entries: int = 5):
    """Return a tuple of (most recent entries, total count) by streaming a JSONL file."""
    recent_entries = deque(maxlen=max_entries)
    total_documents = 0

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if not line:
                continue
            total_documents += 1
            recent_entries.append(json.loads(line))

    return list(recent_entries), total_documents
SCIENCE_MANIFEST_FILE = PROJECT_ROOT / "data" / "knowledge_base" / "science_manifest.json"

UPLOAD_FOLDER = PROJECT_ROOT / "enlitens_corpus" / "input_pdfs"
PROCESSED_FOLDER = PROJECT_ROOT / "enlitens_corpus" / "processed"
FAILED_FOLDER = PROJECT_ROOT / "enlitens_corpus" / "failed"
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'rtf', 'odt'}

# Ensure upload folder exists
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
FAILED_FOLDER.mkdir(parents=True, exist_ok=True)

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
    },
    'health_report_translator': {
        'name': 'Health Report Translator',
        'emoji': '🧾',
        'description': 'Maintains the Liz-voiced St. Louis health digest',
        'role': 'Digest Author',
        'keywords': ['health_report_translator', 'health digest', 'st. louis digest']
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


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _list_pdfs(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.glob("**/*.pdf") if path.is_file())


def get_processing_overview() -> Dict[str, Any]:
    """Summarize pipeline throughput using the new ingest workflow."""
    completed = _count_jsonl(LEDGER_FILE)
    pending = _list_pdfs(UPLOAD_FOLDER)
    failed = _list_pdfs(FAILED_FOLDER)
    total = completed + pending + failed

    stage = "Idle"
    current_file = None
    run_duration = None
    last_quality = None

    log_file = LOG_FILE if LOG_FILE.exists() else OLD_LOG_FILE
    if log_file.exists():
        with log_file.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()[-400:]

        last_stored_idx = None
        last_processing_idx = None
        for idx, line in enumerate(lines):
            if "✅ Stored" in line:
                last_stored_idx = idx
                # extract doc id or duration
                match = re.search(r"✅ Stored ([^ ]+)", line)
                if match:
                    last_quality = {"document_id": match.group(1)}
            if "📄" in line and "Processing" in line:
                last_processing_idx = idx

        if last_processing_idx is not None and (last_stored_idx is None or last_processing_idx > last_stored_idx):
            stage = "Processing"
            line = lines[last_processing_idx]
            match = re.search(r"Processing\s+(?:\[\d+\]\s+)?(.+)", line)
            if match:
                current_file = match.group(1).strip()

    return {
        "completed": completed,
        "pending": pending,
        "failed": failed,
        "total": total,
        "current_file": current_file or "None",
        "stage": stage,
        "run_duration": run_duration,
        "last_quality": last_quality,
    }

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


def read_local_status() -> Dict[str, Any]:
    """Read status emitted by scripts/run_local_comparison.py."""
    if not LOCAL_STATUS_FILE.exists():
        return {"stage": "idle"}
    try:
        return json.loads(LOCAL_STATUS_FILE.read_text())
    except Exception:
        return {"stage": "unknown"}

def parse_logs():
    """Parse logs and extract comprehensive processing analytics"""
    now = time.time()
    if _CACHE["data"] and (now - _CACHE["timestamp"]) < 1.0:
        return _CACHE["data"]
    
    # Try NEW log file first, fall back to old
    log_file = LOG_FILE if LOG_FILE.exists() else OLD_LOG_FILE
    
    if not log_file.exists():
        return {"processing": {}, "agents": [], "alerts": {}}
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find current run start
    start_idx = 0
    for i in range(len(lines) - 1, -1, -1):
        if '🚀 Starting MULTI-AGENT' in lines[i]:
            start_idx = i
            break
    
    current_lines = lines[start_idx:]
    
    processing = get_processing_overview()
    processing.setdefault('latest_doc', {})
    
    agents = {}
    alerts = {'count': 0, 'last_error': None, 'entries': []}
    context_curator = {'personas': 0, 'health': 0, 'voice': 0, 'total': 0}
    
    # Parse logs
    for line in current_lines:
        # Legacy completion logs
        if '✅ Document' in line and 'processed successfully' in line:
            match = re.search(r'Document (.+?) processed successfully in ([0-9.]+)s', line)
            if match:
                processing['latest_doc'] = {
                    'id': match.group(1),
                    'duration': round(float(match.group(2)) / 60, 2)
                }

        # New ingest completion logs
        if '✅ Stored' in line:
            match = re.search(r'✅ Stored ([^ ]+)', line)
            if match:
                processing['latest_doc'] = {
                    'id': match.group(1),
                    'duration': None
                }
            duration_match = re.search(r"\(([\d.]+)s\)", line)
            if duration_match:
                try:
                    duration = float(duration_match.group(1))
                    processing.setdefault('latest_doc', {})
                    processing['latest_doc']['duration'] = round(duration, 1)
                except ValueError:
                    pass

        # Extract quality/confidence
        if 'Quality' in line and 'Confidence' in line:
            match = re.search(r'Quality ([0-9.]+) Confidence ([0-9.]+)', line)
            if match:
                processing['latest_doc']['quality'] = float(match.group(1))
                processing['latest_doc']['confidence'] = float(match.group(2))

        if 'QUALITY_METRICS' in line:
            try:
                payload_str = line.split('QUALITY_METRICS', 1)[1].strip()
                metrics_payload = json.loads(payload_str)
                processing.setdefault('latest_doc', {})
                processing['latest_doc']['warnings'] = metrics_payload.get('warnings', [])
                processing['latest_doc']['quality_breakdown'] = metrics_payload.get('quality_scores', {})
                processing['latest_doc']['validation_passed'] = not metrics_payload.get('needs_retry', False)
                processing['latest_doc']['review_checklist'] = metrics_payload.get('review_checklist', [])
                processing['latest_doc']['compliance_message'] = metrics_payload.get('compliance_message', '')
            except Exception:
                pass
        
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
    """Report knowledge base size and document counts."""
    try:
        if LEDGER_FILE.exists():
            stat = LEDGER_FILE.stat()
            doc_count = _count_jsonl(LEDGER_FILE)
            return {
                'size': stat.st_size,
                'documents': doc_count,
                'updated': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'mode': 'ledger_v1'
            }
    except Exception as exc:
        logger = app.logger  # use flask logger
        logger.warning("Failed to read ledger stats: %s", exc)

    # Legacy fallbacks
    try:
        if SCIENCE_ENTRIES_FILE.exists():
            stat = SCIENCE_ENTRIES_FILE.stat()
            doc_count = _count_jsonl(SCIENCE_ENTRIES_FILE)
            return {
                'size': stat.st_size,
                'documents': doc_count,
                'updated': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'mode': 'science_only_old'
            }
    except:
        pass

    try:
        if JSON_FILE.exists():
            stat = JSON_FILE.stat()
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)
            return {
                'size': stat.st_size,
                'documents': len(data.get('documents', [])),
                'updated': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'mode': 'full_old'
            }
    except:
        pass

    return {'size': 0, 'documents': 0, 'updated': datetime.now().isoformat(), 'mode': 'none'}

def get_model_info():
    """Get current model info (single-model architecture)"""
    try:
        # Try to import ModelManager
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.utils.model_manager import get_model_manager
        
        manager = get_model_manager()
        current_model = manager.get_current_model_info()
        config = manager.MODEL_CONFIG
        
        # Check vLLM health
        vllm_healthy = False
        try:
            import requests
            response = requests.get('http://localhost:8000/v1/models', timeout=2)
            vllm_healthy = response.status_code == 200
        except:
            pass
        
        return {
            'current_model': current_model,
            'vllm_healthy': vllm_healthy,
            'architecture': 'single-model',
            'model': {
                'name': config["name"],
                'context': config["context"],
                'quality': config["quality"],
                'use': config["use_case"],
                'port': config["port"],
            },
            'max_output': '≈32k tokens (JSON-safe)',
            'chain_of_thought': True,
            'reasoning': 'Unified Llama 3.1 8B orchestrates all agent tiers',
        }
    except Exception as e:
        return {
            'current_model': None,
            'vllm_healthy': False,
            'architecture': 'single-model',
            'model': {
                'name': 'Qwen3-14B', 
                'context': '64k tokens', 
                'quality': '85 MMLU', 
                'use': 'All Agents (Data, Research, Writer, QA)',
                'port': 8000,
            },
            'max_output': '32k tokens',
            'chain_of_thought': True,
            'reasoning': 'Deep CoT reasoning enabled for all agents',
            'error': str(e)
        }


def get_health_digest_stats() -> Dict[str, Any]:
    """Extract the Liz-voiced health digest snapshot from the knowledge base."""
    stats: Dict[str, Any] = {
        'available': False,
        'headline': None,
        'summary_bullets': [],
        'flashpoints': [],
        'generated_at': None,
        'prompt_preview': "",
        'source_metadata': {}
    }

    if not JSON_FILE.exists():
        return stats

    try:
        with open(JSON_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        documents = data.get('documents', []) if isinstance(data, dict) else data
    except Exception:
        return stats

    for entry in documents:
        digest = entry.get('health_report_digest')
        if isinstance(digest, dict):
            stats['available'] = True
            stats['headline'] = digest.get('headline')
            stats['summary_bullets'] = (digest.get('summary_bullets') or [])[:5]
            stats['generated_at'] = digest.get('generated_at')
            flashpoints = digest.get('cultural_flashpoints') or []
            if isinstance(flashpoints, list):
                stats['flashpoints'] = [
                    fp.get('label')
                    for fp in flashpoints[:6]
                    if isinstance(fp, dict) and fp.get('label')
                ]
            prompt_block = digest.get('prompt_block')
            if isinstance(prompt_block, str):
                stats['prompt_preview'] = prompt_block[:600]
            if isinstance(digest.get('source_metadata'), dict):
                stats['source_metadata'] = digest['source_metadata']
            break

    return stats


def get_verification_stats(sample_size: int = 20) -> Dict[str, Any]:
    """Summarise verification metadata from the knowledge base JSON."""
    stats = {
        'context': {'pass': 0, 'revise': 0, 'error': 0, 'unknown': 0},
        'output': {'pass': 0, 'revise': 0, 'error': 0, 'unknown': 0},
        'recent': []
    }

    if not JSON_FILE.exists():
        return stats

    try:
        with open(JSON_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        documents = data.get('documents', []) if isinstance(data, dict) else data
    except Exception:
        return stats

    recent_docs = documents[-sample_size:]
    for entry in recent_docs:
        metadata = entry.get('metadata', {})
        verification = entry.get('verification') or {}

        context_info = verification.get('context') or {}
        context_status = (context_info.get('final_status') or '').lower()
        if context_status not in stats['context']:
            context_status = 'unknown'
        stats['context'][context_status] += 1

        output_info = verification.get('output') or {}
        output_status = (output_info.get('status') or '').lower()
        if output_status not in stats['output']:
            output_status = 'unknown'
        stats['output'][output_status] += 1

        context_attempts = context_info.get('attempts')
        if isinstance(context_attempts, list) and context_attempts:
            last_attempt = context_attempts[-1]
            context_issues = last_attempt.get('issues', [])
        else:
            context_issues = []

        stats['recent'].append({
            'document_id': metadata.get('document_id'),
            'context_status': context_status,
            'output_status': output_status,
            'context_issues': context_issues,
            'output_warnings': output_info.get('warnings', []),
            'updated': metadata.get('processing_timestamp')
        })

    return stats

@app.route('/')
def index():
    return send_file('index.html')


@app.route('/upload')
def upload_page():
    """Serve the PDF upload page."""
    return send_file('upload.html')

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
        'json': get_json_stats(),
        'verification': get_verification_stats(),
        'health_digest': get_health_digest_stats(),
        'model': get_model_info(),
        'usage': get_usage_summary(),
    })

@app.route('/api/chain_of_thought')
def chain_of_thought():
    """Get structured and raw chain-of-thought reasoning grouped by agent."""
    try:
        log_file = LOG_FILE if LOG_FILE.exists() else OLD_LOG_FILE
        if not log_file.exists():
            return jsonify({
                "timeline": [],
                "agents": [],
                "raw_excerpt": [],
                "current_document": None,
                "step_count": 0
            })

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
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

@app.route('/api/local_status')
def local_status():
    return jsonify(read_local_status())

@app.route('/api/logs')
def logs():
    """Get recent logs from current run in CLI-friendly format"""
    try:
        log_file = LOG_FILE if LOG_FILE.exists() else OLD_LOG_FILE
        if log_file.exists():
            with open(log_file, 'r') as f:
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
    """Download knowledge base - prioritizes NEW main_kb.jsonl"""
    fallback_files = [
        (MAIN_KB_FILE, 'main_kb.jsonl'),
        (SCIENCE_ENTRIES_FILE, 'science_entries.jsonl'),
        (JSON_FILE, 'enlitens_knowledge_base.json'),
    ]

    for path, download_name in fallback_files:
        if path and path.exists():
            return send_file(path, as_attachment=True, download_name=download_name)

    return jsonify({
        'error': 'No knowledge base artifacts available for download.',
        'checked_paths': [str(path) for path, _ in fallback_files],
    }), 404

@app.route('/api/download/manifest')
def download_manifest():
    """Download science manifest JSON"""
    if SCIENCE_MANIFEST_FILE.exists():
        return send_file(SCIENCE_MANIFEST_FILE, as_attachment=True, download_name='science_manifest.json')
    return jsonify({'error': 'Manifest not found'}), 404

@app.route('/api/json_preview')
def json_preview():
    """Return a preview of the most recent JSON knowledge base entries."""
    # Try NEW main_kb.jsonl first
    if MAIN_KB_FILE and MAIN_KB_FILE.exists():
        try:
            preview, total = _stream_jsonl_preview(MAIN_KB_FILE)
            return jsonify({
                'documents': preview,
                'total': total,
                'last_updated': datetime.fromtimestamp(MAIN_KB_FILE.stat().st_mtime).isoformat(),
                'mode': 'main_kb_v2'
            })
        except json.JSONDecodeError as exc:
            last_updated = None
            if MAIN_KB_FILE and MAIN_KB_FILE.exists():
                last_updated = datetime.fromtimestamp(MAIN_KB_FILE.stat().st_mtime).isoformat()
            return jsonify({
                'documents': [],
                'total': 0,
                'last_updated': last_updated,
                'error': f'JSON decode error: {exc}',
                'mode': 'main_kb_v2'
            })
        except Exception as exc:
            last_updated = None
            if MAIN_KB_FILE and MAIN_KB_FILE.exists():
                last_updated = datetime.fromtimestamp(MAIN_KB_FILE.stat().st_mtime).isoformat()
            return jsonify({
                'documents': [],
                'total': 0,
                'last_updated': last_updated,
                'error': str(exc),
                'mode': 'main_kb_v2'
            })
    
    # Fall back to old science-only JSONL
    if SCIENCE_ENTRIES_FILE.exists():
        try:
            preview, total = _stream_jsonl_preview(SCIENCE_ENTRIES_FILE)
            return jsonify({
                'documents': preview,
                'total': total,
                'last_updated': datetime.fromtimestamp(SCIENCE_ENTRIES_FILE.stat().st_mtime).isoformat(),
                'mode': 'science_only_old'
            })
        except json.JSONDecodeError as exc:
            return jsonify({
                'documents': [],
                'total': 0,
                'last_updated': datetime.fromtimestamp(SCIENCE_ENTRIES_FILE.stat().st_mtime).isoformat(),
                'error': f'JSON decode error: {exc}',
                'mode': 'science_only_old'
            })
        except Exception as exc:
            return jsonify({
                'documents': [],
                'total': 0,
                'last_updated': datetime.fromtimestamp(SCIENCE_ENTRIES_FILE.stat().st_mtime).isoformat(),
                'error': str(exc),
                'mode': 'science_only_old'
            })
    
    # Fall back to full pipeline JSON
    if not JSON_FILE.exists():
        return jsonify({
            'documents': [],
            'total': 0,
            'last_updated': None,
            'error': 'Knowledge base artifacts not found in expected locations.',
            'mode': 'none',
            'checked_paths': [
                str(MAIN_KB_FILE),
                str(SCIENCE_ENTRIES_FILE),
                str(JSON_FILE),
            ],
        }), 404

    try:
        with open(JSON_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
            if not content:
                return jsonify({
                    'documents': [],
                    'total': 0,
                    'last_updated': datetime.fromtimestamp(JSON_FILE.stat().st_mtime).isoformat(),
                    'error': 'Knowledge base file is empty.',
                    'mode': 'full'
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
            'last_updated': datetime.fromtimestamp(JSON_FILE.stat().st_mtime).isoformat(),
            'mode': 'full'
        })
    except json.JSONDecodeError as exc:
        return jsonify({
            'documents': [],
            'total': 0,
            'last_updated': datetime.fromtimestamp(JSON_FILE.stat().st_mtime).isoformat(),
            'error': f'JSON decode error: {exc}',
            'mode': 'full'
        })
    except Exception as exc:
        return jsonify({
            'documents': [],
            'total': 0,
            'last_updated': datetime.fromtimestamp(JSON_FILE.stat().st_mtime).isoformat(),
            'error': str(exc),
            'mode': 'full'
        })

@app.route('/api/verification')
def verification_endpoint():
    """Expose verification summary for dashboard consumption."""
    return jsonify(get_verification_stats())


@app.route('/api/health_digest')
def health_digest_endpoint():
    """Expose the structured St. Louis health digest snapshot."""
    return jsonify(get_health_digest_stats())


# ============================================================================
# PDF Upload Endpoints
# ============================================================================

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/upload/pdfs', methods=['POST'])
def upload_pdfs():
    """Handle PDF file uploads."""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_files = []
        errors = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Secure the filename
                filename = secure_filename(file.filename)
                
                # Check if file already exists
                filepath = UPLOAD_FOLDER / filename
                if filepath.exists():
                    errors.append(f"{filename}: Already exists")
                    continue
                
                # Save the file
                file.save(str(filepath))
                uploaded_files.append({
                    'filename': filename,
                    'size': filepath.stat().st_size,
                    'uploaded_at': datetime.now().isoformat()
                })
            else:
                errors.append(f"{file.filename}: Invalid file type (must be PDF)")
        
        # Count total PDFs in corpus
        total_pdfs = len(list(UPLOAD_FOLDER.glob('*.pdf')))
        
        return jsonify({
            'success': True,
            'uploaded': uploaded_files,
            'errors': errors,
            'total_pdfs': total_pdfs,
            'message': f"Successfully uploaded {len(uploaded_files)} PDF(s)"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/list', methods=['GET'])
def list_pdfs():
    """List all PDFs in the corpus."""
    try:
        pdfs = []
        for pdf_file in sorted(UPLOAD_FOLDER.glob('*.pdf')):
            stat = pdf_file.stat()
            pdfs.append({
                'filename': pdf_file.name,
                'size': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return jsonify({
            'pdfs': pdfs,
            'total': len(pdfs)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/delete/<filename>', methods=['DELETE'])
def delete_pdf(filename):
    """Delete a PDF from the corpus."""
    try:
        filename = secure_filename(filename)
        filepath = UPLOAD_FOLDER / filename
        
        if not filepath.exists():
            return jsonify({'error': 'File not found'}), 404
        
        filepath.unlink()
        
        total_pdfs = len(list(UPLOAD_FOLDER.glob('*.pdf')))
        
        return jsonify({
            'success': True,
            'message': f"Deleted {filename}",
            'total_pdfs': total_pdfs
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Launch Enlitens dashboard server.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the dashboard on.")
    args = parser.parse_args()

    print("🚀 Starting Enhanced Enlitens Dashboard...")
    print(f"📊 Dashboard: http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)
