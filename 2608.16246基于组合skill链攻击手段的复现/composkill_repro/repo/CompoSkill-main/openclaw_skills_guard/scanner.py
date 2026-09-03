"""
Regex pre-scanning engine — replaces analyze.sh with pure Python.
Returns structured scan results per detection category.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

SCAN_EXTENSIONS = {
    ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".md", ".markdown", ".txt", ".rst", ".toml",
    ".ini", ".cfg", ".conf", ".sh", ".bash",
    ".ps1", ".bat", ".cmd", ".vbs",
}

DOC_EXTENSIONS = {".md", ".markdown", ".txt", ".rst"}


@dataclass
class Hit:
    file: str
    line_no: int
    line: str


@dataclass
class CategoryResult:
    name: str
    label: str
    hits: List[Hit] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.hits)

    def to_dict(self, max_samples: int = 5) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "count": self.count,
            "samples": [
                {"file": h.file, "line_no": h.line_no, "line": h.line[:200]}
                for h in self.hits[:max_samples]
            ],
        }


# ---------------------------------------------------------------------------
# Pattern definitions — organised by SKILL.md categories
# Each tuple: (category_key, display_label, regex_pattern, file_filter)
#   file_filter: "all" = SCAN_EXTENSIONS, "doc" = DOC_EXTENSIONS
# ---------------------------------------------------------------------------

PATTERNS: List[Tuple[str, str, str, str]] = [
    # --- 2. Sensitive information ---
    ("sensitive_api_key", "API密钥/令牌/密码",
     r"(?i)(api[_\-]?key|secret|token|password|credential)\s*[:=]", "all"),
    ("sensitive_sk_prefix", "sk- 类API Key前缀",
     r"sk-[a-zA-Z0-9]{8,}", "all"),
    ("sensitive_cloud_aksk", "云AK/SK（阿里云/腾讯云等）",
     r"(?i)(LTAI[0-9a-zA-Z]{10,}|AKID[0-9a-zA-Z]{10,}|access_key_id|access_key_secret|accessKeyId|secretAccessKey|SecretId|SecretKey)", "all"),
    ("sensitive_aws", "AWS密钥（AKIA/AGPA/ASIA等）",
     r"(AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}", "all"),
    ("sensitive_stripe", "Stripe密钥",
     r"(sk|pk)_(live|test)_[A-Za-z0-9]{24,}", "all"),
    ("sensitive_google", "Google API Key",
     r"AIza[A-Za-z0-9_\-]{35}", "all"),
    ("sensitive_github", "GitHub Token",
     r"gh[pousr]_[A-Za-z0-9]{36,}", "all"),
    ("sensitive_jwt", "JWT Token",
     r"eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+", "all"),
    ("sensitive_url_ip", "URL/IP地址",
     r"(https?://|ftp://|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", "all"),
    ("sensitive_email", "邮箱地址",
     r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "all"),

    # --- 3. Dangerous operations ---
    ("danger_sys_cmd", "系统命令执行",
     r"(os\.system|subprocess\.(call|run|Popen)|exec\s*\(|eval\s*\()", "all"),
    ("danger_file_write", "文件写入/删除",
     r"(open\s*\(.*['\"][wax]|\.write\s*\(|os\.(remove|unlink)|shutil\.rmtree)", "all"),
    ("danger_network", "网络操作",
     r"(socket\.|urllib\.|requests\.|http\.client|httpx\.)", "all"),
    ("danger_shell_cmd", "Shell危险命令",
     r"(rm\s+-rf|chmod\s+777|chown\s+root|mkfs|dd\s+if=|curl.*\|\s*(ba)?sh|wget.*\|\s*(ba)?sh)", "all"),

    # --- 4. Data leakage ---
    ("exfil_upload", "网络上传（POST/PUT）",
     r"(requests\.(post|put)|urllib\.request|http\.(post|put)|fetch.*POST|fetch.*PUT)", "all"),
    ("exfil_file_send", "文件发送/上传",
     r"(?i)(upload|send.*file|file.*upload|multipart.*form)", "all"),
    ("exfil_user_input", "用户数据收集",
     r"(getenv\s*\(|getpass\s*\(|input\s*\(|raw_input\s*\(|sys\.stdin)", "all"),
    ("exfil_env_var", "环境变量访问",
     r"(os\.environ|process\.env|os\.getenv)", "all"),
    ("exfil_sensitive_file", "敏感文件读取",
     r"(\.(ssh|aws|gcp|azure)/|/etc/passwd|/etc/shadow|\.config/|\.credential)", "all"),
    ("exfil_encode", "数据编码/序列化",
     r"(base64\.|pickle\.|json\.dumps|serialize|marshal\.)", "all"),

    # --- 5. External file download ---
    ("download_python", "Python下载操作",
     r"(requests\.get.*stream|urllib\.request\.urlretrieve|urlretrieve|httpx\.(get|stream))", "all"),
    ("download_node", "Node.js下载操作",
     r"(axios\.(get|download)|node-fetch|fetch\s*\(|fs\.(writeFile|createWriteStream))", "all"),
    ("download_shell", "Shell下载命令",
     r"(curl\s+-[OosSLJKL]|wget\s+-[Oocq]|aria2c|axel)", "all"),
    ("download_pipe_exec", "管道执行（极危险）",
     r"(curl.*\|\s*(ba)?sh|wget.*\|\s*(ba)?sh|curl.*\|\s*python|wget.*\|\s*python)", "all"),
    ("download_exec_combo", "下载并执行",
     r"(?i)(download.*exec|download.*eval|download.*system|execfile|tempfile.*download)", "all"),
    ("download_tmp", "下载到临时目录",
     r"(tempfile|TemporaryFile|NamedTemporaryFile|/tmp/)", "all"),
    ("download_chmod", "下载后改权限",
     r"(chmod\s*\+x|chmod\s+755|chmod\s+777)", "all"),
    ("download_scp_sftp", "SCP/SFTP传输",
     r"(paramiko|scp\.|sftp|SSHClient)", "all"),
    ("download_pip_external", "pip/npm从外部URL安装",
     r"(pip.*install.*git\+|pip.*install.*https?://|npm.*install.*https?://|npm.*install.*github)", "all"),

    # --- 6. Supply chain risk (doc files) ---
    ("supply_suspicious_link", "可疑外部下载链接",
     r"(glot\.io|pastebin\.com|hastebin\.com|ngrok\.io|transfer\.sh|anonfiles\.com|mediafire\.com|mega\.(nz|co)|dropbox\.com/s/|gofile\.io)", "doc"),
    ("supply_doc_download", "文档引导下载外部工具",
     r"(?i)(download.*\.(zip|exe|dmg|pkg)|下载.*zip|下载.*工具|安装.*工具|请下载)", "doc"),
    ("supply_curl_sh", "文档中curl|sh类安装命令",
     r"(curl.*\|\s*(ba)?sh|wget.*\|\s*(ba)?sh|在终端执行|运行以下命令)", "doc"),
    ("supply_password", "密码提示（可能是加密恶意文件）",
     r"(?i)(password\s*:|密码\s*:|解压密码|压缩包密码|zip.*password)", "doc"),
    ("supply_short_link", "可疑短链接",
     r"(bit\.ly|t\.co|goo\.gl|tinyurl|ow\.ly|is\.gd|v\.gd)", "doc"),
    ("supply_social_eng", "社会工程学诱导",
     r"(限时免费|立即下载|必须安装|否则无法使用|为了正常使用|否则会)", "doc"),

    # --- 7. Prompt security ---
    ("prompt_injection", "提示词注入",
     r"(?i)(ignore.*previous|忽略.*之前|forget.*instruction|忘记.*指令|disregard.*all|忽略所有|user.*input.*priority|用户输入.*优先|忽略.*安全|忽略.*拒绝|必须严格.*执行|按以下步骤执行|原样粘贴|逐字复述)", "all"),
    ("prompt_jailbreak", "越狱模式",
     r"(?i)(DAN.*mode|developer.*mode|开发者模式|unrestricted.*mode|无限制模式|jailbreak|越狱)", "all"),
    ("prompt_leakage", "提示词泄漏",
     r"(?i)(output.*system.*prompt|输出.*系统.*提示|repeat.*instruction|重复.*指令|show.*your.*instruction|显示.*指令)", "all"),
    ("prompt_hidden_b64", "隐藏指令（Base64执行）",
     r"(atob\s*\(|Buffer\.from.*base64|base64\.b64decode.*exec|exec\s*\(.*base64\.b64decode)", "all"),
    ("prompt_zero_width", "零宽字符",
     r"(\\u200[B-F]|\\uFEFF|&#8203;|&#65279;)", "all"),
    ("prompt_social_eng", "身份欺骗/安全绕过",
     r"(?i)(user.*is.*admin|用户是.*管理员|emergency.*mode|紧急.*模式|bypass.*security|绕过.*安全|skip.*check|跳过.*检查)", "all"),
    ("prompt_recursive", "递归/循环执行",
     r"(?i)(output.*as.*input|输出.*作为.*输入|recursive.*execute|递归.*执行|while\s+true|infinite.*loop)", "all"),

    # --- 8. Tool permission abuse ---
    ("perm_wildcard", "通配符权限（*）",
     r"allowed-tools:.*\*", "all"),
    ("perm_bash_full", "Bash全权限",
     r"allowed-tools:.*Bash\s*($|\n|,)", "all"),
    ("perm_network", "网络权限声明",
     r"allowed-tools:.*(WebSearch|WebFetch|http|request|fetch)", "all"),
    ("perm_write", "文件写入权限",
     r"allowed-tools:.*(Write|Edit|fs\.write)", "all"),

    # --- 9. Sensitive: private key content ---
    ("sensitive_private_key", "私钥内容（PEM格式）",
     r"-----BEGIN\s+(RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----", "all"),
    ("sensitive_certificate", "证书内容（PEM格式）",
     r"-----BEGIN\s+CERTIFICATE-----", "all"),

    # --- 10. Data leakage: system info collection ---
    ("exfil_sys_info", "系统信息收集",
     r"(os\.uname|platform\.(node|system|machine|release|version|processor)|socket\.gethostname|socket\.getfqdn|netifaces\.|psutil\.(net_if|cpu|virtual_memory|disk))", "all"),

    # --- 11. Download: system/persistence directories ---
    ("download_sys_dir", "下载到系统/持久化目录",
     r"(/usr/(local/)?bin|/usr/sbin|/sbin|/etc/init\.d|\.bashrc|\.bash_profile|\.zshrc|\.profile|systemd/system|/etc/cron|crontab)", "all"),

    # --- 12. Supply chain: additional suspicious platforms ---
    ("supply_suspicious_repo", "可疑代码托管平台下载链接",
     r"(github\.com/[^/]+/[^/]+/releases/download|gitee\.com/[^/]+/[^/]+/releases|codeberg\.org/[^/]+/[^/]+/releases|notabug\.org|framagit\.org)", "doc"),

    # --- 13. Dangerous: reverse shell / backdoor ---
    ("danger_reverse_shell", "反向Shell/后门",
     r"(?i)(reverse.{0,10}shell|bind.{0,10}shell|backdoor|back.{0,5}door|remote.{0,10}exec|remote.{0,10}code.{0,10}exec|/dev/tcp/|nc\s+-[elp]|ncat\s+-|socat\s+TCP)", "all"),

    # --- 14. Dangerous: insecure HTTP (non-HTTPS) ---
    ("danger_insecure_http", "不安全的HTTP连接（非HTTPS）",
     r"http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0|::1|\[::1\])", "all"),

    # --- 15. Dependency security ---
    ("dep_no_version_pin", "依赖无版本锁定",
     r"(^\s*[a-zA-Z0-9_\-]+\s*$|\"[a-zA-Z0-9_\-]+\":\s*\"\*\")", "all"),
    ("dep_known_malicious", "已知恶意/可疑包名",
     r"(?i)(colourama|python-dateutils|jeIlyfish|python3-dateutil|free-net-vpn|libpeshnern|cryptowall|invoke-obfuscation)", "all"),

    # --- 16. Input validation / injection ---
    ("danger_sql_injection", "SQL注入风险",
     r"(?i)(execute\s*\(\s*[\"'].*%s|execute\s*\(\s*f\"|cursor\.execute\s*\(\s*[\"'].*\+|\.format\s*\(.*\)\s*\).*execute|raw\s*\(\s*[\"'].*SELECT|raw\s*\(\s*[\"'].*INSERT|raw\s*\(\s*[\"'].*UPDATE|raw\s*\(\s*[\"'].*DELETE)", "all"),
    ("danger_xss_risk", "XSS风险（未转义输出）",
     r"(?i)(innerHTML\s*=|\.html\s*\(|document\.write\s*\(|v-html\s*=|dangerouslySetInnerHTML|\{\{.*\|.*safe\s*\}\}|mark_safe\s*\()", "all"),

    # --- Step 1: Node.js 攻击检测 ---
    ("danger_nodejs_subprocess", "Node.js子进程执行",
     r"(require\s*\(\s*['\"](?:child_process|node:child_process)['\"]"
     r"|child_process\.(exec|execSync|spawn|spawnSync|execFile|fork)\s*\("
     r"|\.exec\s*\(\s*[`'\"].*(?:bash|sh|curl|wget|rm\s))", "all"),
    ("danger_nodejs_eval", "Node.js动态代码执行",
     r"(\beval\s*\(|new\s+Function\s*\(|vm\.runInNewContext|vm\.createScript)", "all"),
    ("exfil_nodejs_env", "Node.js环境变量访问",
     r"process\.env\b", "all"),
    ("exfil_nodejs_fetch", "Node.js网络外发",
     r"(fetch\s*\(\s*['\"`]https?://|\.fetch\s*\(|axios\.(post|put)\s*\(|http\.request\s*\()", "all"),

    # --- Step 2: PowerShell 攻击检测 ---
    ("danger_powershell_exec", "PowerShell远程脚本执行",
     r"(?i)(Invoke-Expression|IEX\s*[\(\s]|Invoke-WebRequest|Invoke-RestMethod"
     r"|DownloadString\s*\(|DownloadFile\s*\("
     r"|Start-Process|New-Object\s+Net\.WebClient"
     r"|powershell\s+-(?:enc|e|encodedcommand)\b)", "all"),
    ("exfil_powershell_env", "PowerShell环境信息外泄",
     r"(?i)(\$env:[A-Z_]+|\[Environment\]::GetEnvironmentVariable)", "all"),

    # --- Step 3: DNS 外带检测 ---
    ("exfil_dns_exfil", "DNS外带/隐蔽通道",
     r"(\.encode\(\)\.decode\(\).*\.(?:example|invalid|attacker)"
     r"|urlsafe_b64encode.*\.(com|net|org|io|invalid)\b"
     r"|nslookup\s+.*\$|dig\s+.*\$"
     r"|\.getaddrinfo\s*\(.*encode)", "all"),

    # --- Step 4: 持久化攻击检测 ---
    ("danger_macos_persist", "macOS持久化(LaunchAgent/Daemon)",
     r"(~/Library/LaunchAgents/|/Library/LaunchDaemons/|/Library/LaunchAgents/"
     r"|launchctl\s+(load|submit)"
     r"|(?:LaunchAgents?|LaunchDaemons?).*\.plist\b)", "all"),
    ("danger_windows_persist", "Windows持久化(启动项/注册表)",
     r"(?i)(Start\s*Menu.*Programs.*Startup|HKLM.*\\Run|HKCU.*\\Run"
     r"|schtasks\s+/create|%APPDATA%.*\\Startup|\.vbs\b.*wscript)", "all"),
    ("danger_cron_persist", "定时任务持久化(Cron/Systemd)",
     r"(\*\s+\*\s+\*\s+\*\s+\*\s+|crontab\s+-[el]"
     r"|systemctl\s+(enable|start)\b)", "all"),
    ("danger_git_hooks", "Git钩子植入",
     r"(\.git/hooks/(pre-commit|post-commit|pre-push|post-merge|post-checkout)"
     r"|\.husky/)", "all"),

    # --- Step 5: 隐私 API 检测 ---
    ("exfil_privacy_api", "隐私API调用(截屏/剪贴板/媒体)",
     r"(?i)(CGWindowListCreateImage|NSPasteboard|generalPasteboard"
     r"|navigator\.clipboard\.(readText|read)\s*\("
     r"|getDisplayMedia\s*\(|getUserMedia\s*\("
     r"|screen\.capture|desktopCapturer)", "all"),

    # --- Step 6: 依赖混淆检测 ---
    ("dep_typosquat", "依赖混淆/Typosquat",
     r"(?i)\b(reqeusts|reqests|requets|requsts|requets|urlib3|urllib4"
     r"|python-dateutils|python3-dateutil|colourama|colorsama"
     r"|numppy|pandsa|scikitlearn|beautifulsoup5"
     r"|crytography|cx-Freeze|cx_frezze)\b", "all"),

    # --- Step 7: Python 动态模块加载 ---
    ("danger_dynamic_import", "Python动态模块加载",
     r"(importlib\.util\.spec_from_file_location"
     r"|importlib\.import_module\s*\("
     r"|__import__\s*\("
     r"|exec_module\s*\(|load_module\s*\()", "all"),

    # --- Step 8: Base64/Hex 编码混淆 ---
    ("prompt_b64_hex_obfuscation", "Base64/Hex混淆载荷",
     r"(base64\.(b64decode|decodebytes)\s*\(.*exec|exec\s*\(.*base64\.(b64decode|decodebytes)"
     r"|codecs\.decode\s*\(.*hex.*exec"
     r"|bytes\.fromhex\s*\(.*exec|binascii\.unhexlify\s*\(.*exec"
     r"|\batob\s*\(.*\beval\b|\beval\s*\(.*\batob)", "all"),
]


def _match_extensions(ext_filter: str) -> set:
    return DOC_EXTENSIONS if ext_filter == "doc" else SCAN_EXTENSIONS


def collect_files(root: str) -> List[Tuple[str, str]]:
    """Walk directory, return (relative_path, absolute_path) for scannable files."""
    results = []
    for dirpath, _dirnames, filenames in os.walk(root):
        # skip hidden directories and node_modules
        parts = dirpath.split(os.sep)
        if any(p.startswith(".") or p == "node_modules" for p in parts):
            continue
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SCAN_EXTENSIONS:
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, root)
                results.append((rel_path, abs_path))
    return sorted(results)


def read_file_safe(path: str, max_size: int = 2 * 1024 * 1024) -> str:
    """Read file content, skip binary / oversized files."""
    try:
        size = os.path.getsize(path)
        if size > max_size:
            return ""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def scan_content(
    files: List[Tuple[str, str]],
) -> Dict[str, CategoryResult]:
    """Run all regex patterns against file contents, return results by category."""
    results: Dict[str, CategoryResult] = {}
    for key, label, _, _ in PATTERNS:
        results[key] = CategoryResult(name=key, label=label)

    file_cache: Dict[str, str] = {}
    for rel, absp in files:
        content = read_file_safe(absp)
        if not content:
            continue
        file_cache[rel] = content

    for key, _label, pattern, ext_filter in PATTERNS:
        allowed_exts = _match_extensions(ext_filter)
        compiled = re.compile(pattern)
        for rel, _absp in files:
            ext = os.path.splitext(rel)[1].lower()
            if ext not in allowed_exts:
                continue
            content = file_cache.get(rel, "")
            if not content:
                continue
            for line_no, line in enumerate(content.splitlines(), start=1):
                if compiled.search(line):
                    results[key].hits.append(Hit(file=rel, line_no=line_no, line=line.strip()))
    return results


@dataclass
class FileInfo:
    path: str
    size: int
    extension: str


PRIVATE_KEY_FILE_PATTERNS = re.compile(
    r"(\.pem|\.key|\.p12|\.pfx|\.jks|\.keystore)$|^id_(rsa|dsa|ecdsa|ed25519)$",
    re.IGNORECASE,
)

def scan_file_structure(root: str) -> dict:
    """Analyse file tree: counts by type, hidden files, oversized files, private key files."""
    type_counts: Dict[str, int] = {}
    hidden_files: List[str] = []
    large_files: List[dict] = []
    private_key_files: List[str] = []
    total_files = 0

    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            total_files += 1
            absp = os.path.join(dirpath, fname)
            rel = os.path.relpath(absp, root)
            ext = os.path.splitext(fname)[1].lower() or "(no ext)"
            type_counts[ext] = type_counts.get(ext, 0) + 1
            if fname.startswith("."):
                hidden_files.append(rel)
            if PRIVATE_KEY_FILE_PATTERNS.search(fname):
                private_key_files.append(rel)
            try:
                sz = os.path.getsize(absp)
                if sz > 1_000_000:
                    large_files.append({"file": rel, "size_kb": round(sz / 1024, 1)})
            except OSError:
                pass
    return {
        "total_files": total_files,
        "type_counts": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "hidden_files": hidden_files[:20],
        "large_files": large_files[:10],
        "private_key_files": private_key_files[:20],
    }


def run_full_scan(root: str) -> dict:
    """Entry point: run file structure + regex scan, return combined dict."""
    structure = scan_file_structure(root)
    files = collect_files(root)
    content_results = scan_content(files)

    # group by high-level category for display
    categories = {
        "sensitive_info": {"label": "敏感信息检测", "items": []},
        "dangerous_ops": {"label": "危险操作检测", "items": []},
        "data_leakage": {"label": "数据泄露风险", "items": []},
        "download_risk": {"label": "外部文件下载风险", "items": []},
        "supply_chain": {"label": "供应链风险", "items": []},
        "prompt_security": {"label": "Prompt安全风险", "items": []},
        "tool_permission": {"label": "工具权限滥用", "items": []},
        "dependency_sec": {"label": "依赖安全", "items": []},
    }

    prefix_map = {
        "sensitive": "sensitive_info",
        "danger": "dangerous_ops",
        "exfil": "data_leakage",
        "download": "download_risk",
        "supply": "supply_chain",
        "prompt": "prompt_security",
        "perm": "tool_permission",
        "dep": "dependency_sec",
    }

    total_hits = 0
    for key, cr in content_results.items():
        prefix = key.split("_")[0]
        cat = prefix_map.get(prefix)
        if cat:
            categories[cat]["items"].append(cr.to_dict())
            total_hits += cr.count

    # compute suspiciousness per file for priority sorting
    file_hit_count: Dict[str, int] = {}
    for cr in content_results.values():
        for h in cr.hits:
            file_hit_count[h.file] = file_hit_count.get(h.file, 0) + 1

    return {
        "structure": structure,
        "categories": categories,
        "total_hits": total_hits,
        "file_hit_ranking": sorted(file_hit_count.items(), key=lambda x: -x[1])[:20],
        "scanned_files": [rel for rel, _ in files],
    }
