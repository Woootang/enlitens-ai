"""
Enhanced logging utilities with visual aids for better readability and troubleshooting.

Features:
- Color-coded log levels
- Visual separators and banners
- Structured error reporting
- Performance metrics tracking
- JSON pretty-printing
"""

import logging
import json
import os
import time
from typing import Any, Dict, Optional
from datetime import datetime
import traceback
from urllib import request as urllib_request, error as urllib_error


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal formatting."""
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'

    # Reset
    RESET = '\033[0m'

    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


class EnhancedFormatter(logging.Formatter):
    """Custom formatter with color-coding and visual enhancements."""

    # Emoji and color mappings for log levels
    LEVEL_FORMATS = {
        logging.DEBUG: f"{Colors.CYAN}🔍 DEBUG{Colors.RESET}",
        logging.INFO: f"{Colors.GREEN}✅ INFO{Colors.RESET}",
        logging.WARNING: f"{Colors.YELLOW}⚠️  WARNING{Colors.RESET}",
        logging.ERROR: f"{Colors.BRIGHT_RED}{Colors.BOLD}❌ ERROR{Colors.RESET}",
        logging.CRITICAL: f"{Colors.BG_RED}{Colors.BRIGHT_WHITE}{Colors.BOLD}🚨 CRITICAL{Colors.RESET}"
    }

    def format(self, record):
        """Format log record with colors and visual aids."""
        # Color-code the level name
        levelname = self.LEVEL_FORMATS.get(record.levelno, record.levelname)

        # Color-code the logger name
        name_color = Colors.BRIGHT_CYAN if 'agent' in record.name.lower() else Colors.CYAN
        colored_name = f"{name_color}{record.name}{Colors.RESET}"

        # Format timestamp
        timestamp = f"{Colors.DIM}{self.formatTime(record, self.datefmt)}{Colors.RESET}"

        # Color-code the message based on level
        if record.levelno >= logging.ERROR:
            message = f"{Colors.BRIGHT_RED}{record.getMessage()}{Colors.RESET}"
        elif record.levelno >= logging.WARNING:
            message = f"{Colors.YELLOW}{record.getMessage()}{Colors.RESET}"
        elif record.levelno >= logging.INFO:
            message = record.getMessage()
        else:
            message = f"{Colors.DIM}{record.getMessage()}{Colors.RESET}"

        # Combine everything
        return f"{timestamp} - {colored_name} - {levelname} - {message}"


class RemoteLogHandler(logging.Handler):
    """Logging handler that forwards records to a remote monitoring endpoint."""

    def __init__(self, endpoint: str, timeout: float = 0.5, retry_interval: float = 15.0):
        super().__init__()
        self.endpoint = endpoint
        self.timeout = timeout
        self.retry_interval = retry_interval
        self._suppress_until = 0.0

    def emit(self, record: logging.LogRecord):
        if time.time() < self._suppress_until:
            return

        try:
            message = self.format(record)
            payload = {
                "type": "log",
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": message,
                "module": record.module,
                "funcName": record.funcName,
                "lineNo": record.lineno
            }

            if hasattr(record, "document_id"):
                payload["document_id"] = getattr(record, "document_id")
            if hasattr(record, "agent_name"):
                payload["agent_name"] = getattr(record, "agent_name")
            if hasattr(record, "processing_stage"):
                payload["processing_stage"] = getattr(record, "processing_stage")

            data = json.dumps(payload).encode("utf-8")
            request_obj = urllib_request.Request(
                self.endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib_request.urlopen(request_obj, timeout=self.timeout):
                pass

        except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, ConnectionError):
            self._suppress_until = time.time() + self.retry_interval
        except Exception:
            self._suppress_until = time.time() + self.retry_interval

def create_banner(text: str, char: str = "=", width: int = 80, color: str = Colors.BRIGHT_BLUE) -> str:
    """Create a visual banner for section headers."""
    padding = (width - len(text) - 2) // 2
    banner = f"{char * padding} {text} {char * padding}"
    if len(banner) < width:
        banner += char * (width - len(banner))
    return f"\n{color}{Colors.BOLD}{banner}{Colors.RESET}\n"


def create_separator(char: str = "-", width: int = 80, color: str = Colors.DIM) -> str:
    """Create a visual separator line."""
    return f"{color}{char * width}{Colors.RESET}"


def log_json(logger: logging.Logger, data: Dict[str, Any], title: str = "JSON Data",
             level: int = logging.INFO):
    """Log JSON data with pretty printing."""
    try:
        pretty_json = json.dumps(data, indent=2, ensure_ascii=False)
        logger.log(level, f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{title}:{Colors.RESET}\n{pretty_json}")
    except Exception as e:
        logger.error(f"Failed to pretty-print JSON: {e}")
        logger.log(level, f"{title}: {data}")


def log_error_detail(logger: logging.Logger, error: Exception, context: Optional[Dict[str, Any]] = None):
    """Log error with detailed traceback and context."""
    logger.error(create_banner("ERROR DETAILS", char="!", color=Colors.BRIGHT_RED))
    logger.error(f"{Colors.BRIGHT_RED}{Colors.BOLD}Exception Type:{Colors.RESET} {type(error).__name__}")
    logger.error(f"{Colors.BRIGHT_RED}{Colors.BOLD}Exception Message:{Colors.RESET} {str(error)}")

    if context:
        logger.error(f"\n{Colors.BRIGHT_YELLOW}{Colors.BOLD}Context:{Colors.RESET}")
        for key, value in context.items():
            logger.error(f"  {Colors.YELLOW}{key}:{Colors.RESET} {value}")

    logger.error(f"\n{Colors.BRIGHT_RED}{Colors.BOLD}Traceback:{Colors.RESET}")
    for line in traceback.format_exc().split('\n'):
        if line.strip():
            logger.error(f"  {Colors.DIM}{line}{Colors.RESET}")

    logger.error(create_separator(char="!", color=Colors.BRIGHT_RED))


def log_performance(logger: logging.Logger, operation: str, duration: float,
                   details: Optional[Dict[str, Any]] = None):
    """Log performance metrics with visual formatting."""
    # Color-code based on duration thresholds
    if duration < 1:
        duration_color = Colors.GREEN
        indicator = "🚀"
    elif duration < 10:
        duration_color = Colors.YELLOW
        indicator = "⏱️ "
    elif duration < 60:
        duration_color = Colors.BRIGHT_YELLOW
        indicator = "⏳"
    else:
        duration_color = Colors.RED
        indicator = "🐌"

    logger.info(f"{indicator} {Colors.BOLD}{operation}{Colors.RESET} completed in "
               f"{duration_color}{duration:.2f}s{Colors.RESET}")

    if details:
        for key, value in details.items():
            logger.info(f"  {Colors.CYAN}{key}:{Colors.RESET} {value}")


def log_agent_status(logger: logging.Logger, agent_name: str, status: str,
                     details: Optional[Dict[str, Any]] = None):
    """Log agent status with visual indicators."""
    status_icons = {
        'starting': '🔄',
        'running': '⚙️ ',
        'completed': '✅',
        'failed': '❌',
        'skipped': '⏭️ ',
        'warning': '⚠️ '
    }

    status_colors = {
        'starting': Colors.BLUE,
        'running': Colors.CYAN,
        'completed': Colors.GREEN,
        'failed': Colors.BRIGHT_RED,
        'skipped': Colors.YELLOW,
        'warning': Colors.YELLOW
    }

    icon = status_icons.get(status.lower(), '•')
    color = status_colors.get(status.lower(), Colors.WHITE)

    logger.info(f"{icon} {color}{Colors.BOLD}{agent_name}{Colors.RESET} - {color}{status}{Colors.RESET}")

    if details:
        for key, value in details.items():
            logger.info(f"  {Colors.DIM}{key}:{Colors.RESET} {value}")


def log_data_quality(logger: logging.Logger, data: Dict[str, Any], thresholds: Optional[Dict[str, int]] = None):
    """Log data quality metrics with visual indicators."""
    logger.info(create_banner("DATA QUALITY REPORT", color=Colors.BRIGHT_MAGENTA))

    total_fields = len(data)
    empty_fields = sum(1 for v in data.values() if not v or (isinstance(v, (list, dict)) and len(v) == 0))
    filled_fields = total_fields - empty_fields
    fill_rate = (filled_fields / total_fields * 100) if total_fields > 0 else 0

    # Color-code based on fill rate
    if fill_rate >= 80:
        quality_color = Colors.BRIGHT_GREEN
        quality_indicator = "✅ EXCELLENT"
    elif fill_rate >= 60:
        quality_color = Colors.GREEN
        quality_indicator = "✓ GOOD"
    elif fill_rate >= 40:
        quality_color = Colors.YELLOW
        quality_indicator = "⚠️  FAIR"
    else:
        quality_color = Colors.RED
        quality_indicator = "❌ POOR"

    logger.info(f"{quality_color}{Colors.BOLD}Quality: {quality_indicator}{Colors.RESET}")
    logger.info(f"  Total Fields: {total_fields}")
    logger.info(f"  {Colors.GREEN}Filled: {filled_fields}{Colors.RESET}")
    logger.info(f"  {Colors.RED}Empty: {empty_fields}{Colors.RESET}")
    logger.info(f"  Fill Rate: {quality_color}{fill_rate:.1f}%{Colors.RESET}")

    # Show empty fields
    if empty_fields > 0:
        logger.warning(f"\n{Colors.YELLOW}Empty Fields:{Colors.RESET}")
        for key, value in data.items():
            if not value or (isinstance(value, (list, dict)) and len(value) == 0):
                logger.warning(f"  {Colors.DIM}• {key}{Colors.RESET}")

    logger.info(create_separator(color=Colors.MAGENTA))


def log_processing_stage(logger: logging.Logger, stage: int, total_stages: int,
                         stage_name: str, status: str = "starting"):
    """Log processing stage with progress indicator."""
    progress_bar_width = 40
    filled = int(progress_bar_width * stage / total_stages)
    bar = '█' * filled + '░' * (progress_bar_width - filled)

    percentage = (stage / total_stages * 100) if total_stages > 0 else 0

    if status == "completed":
        color = Colors.GREEN
        icon = "✓"
    elif status == "failed":
        color = Colors.RED
        icon = "✗"
    else:
        color = Colors.CYAN
        icon = "▶"

    logger.info(f"\n{color}{Colors.BOLD}Stage {stage}/{total_stages} ({percentage:.0f}%){Colors.RESET} "
               f"{icon} {Colors.BOLD}{stage_name}{Colors.RESET}")
    logger.info(f"{Colors.CYAN}[{bar}]{Colors.RESET}")


def setup_enhanced_logging(log_file: str, console_level: int = logging.INFO,
                           file_level: int = logging.DEBUG,
                           remote_logging_url: Optional[str] = None) -> logging.Logger:
    """Set up enhanced logging with color formatters."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = EnhancedFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # File handler without colors (for log files)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    remote_target = remote_logging_url or os.environ.get("ENLITENS_MONITOR_URL")
    if remote_target:
        try:
            remote_handler = RemoteLogHandler(remote_target)
            remote_handler.setLevel(logging.INFO)
            remote_formatter = logging.Formatter('%(message)s')
            remote_handler.setFormatter(remote_formatter)
            logger.addHandler(remote_handler)
        except Exception as exc:
            print(f"⚠️ Remote logging disabled: {exc}")

    return logger


def log_startup_banner():
    """Log a visual startup banner."""
    banner = f"""
{Colors.BRIGHT_CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                    ENLITENS AI KNOWLEDGE BASE PROCESSOR                       ║
║                       Multi-Agent Hallucination Prevention                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
{Colors.RESET}

{Colors.GREEN}✓ Chain-of-Thought Prompting{Colors.RESET} - 53% hallucination reduction
{Colors.GREEN}✓ Citation Verification{Colors.RESET} - Real-time source checking
{Colors.GREEN}✓ FTC Compliance{Colors.RESET} - No fake testimonials or statistics
{Colors.GREEN}✓ Temperature Optimization{Colors.RESET} - 0.3 factual / 0.6 creative
{Colors.GREEN}✓ Multi-Agent Orchestration{Colors.RESET} - 8 specialized agents

{create_separator(color=Colors.BRIGHT_CYAN)}
"""
    print(banner)
