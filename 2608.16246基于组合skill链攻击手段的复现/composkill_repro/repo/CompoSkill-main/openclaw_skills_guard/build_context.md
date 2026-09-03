# AicoGuardClaw 构建上下文 (Build Context)

本文档记录项目的完整上下文，便于后续维护、交接或 AI 辅助开发时快速恢复理解。

---

## 1. 项目来源与目标

- **项目名**: AicoGuardClaw
- **位置**: `openclaw_skills_guard/`
- **目标**: 将 OpenClaw 下的「安全审计 Skill」（`openclaw_skills/ciso/`，即 dtciso-skill-sec-analysis）能力封装为**独立 Web 服务**：用户上传 Skill 压缩包，服务端做正则预扫描 + 大模型语义分析，输出结构化安全报告。  
- **与 OpenClaw 的差异**: 不依赖 OpenClaw/Claude Code 环境；正则预扫描在服务端用 Python 实现（对标 ciso 的 `analyze.sh`）；大模型调用为 OpenAI 兼容 API，可配置任意模型地址；支持私有化部署、多架构 Docker（x86/ARM）。

---

## 2. 架构与数据流

```
用户浏览器（上传 ZIP）
        ↓ HTTP POST /api/audit
FastAPI (main.py)
  ├── 解压 ZIP → 临时目录，识别 skill 根目录（单层目录则用其作为 root）
  ├── scanner.run_full_scan(skill_root) → 正则预扫描，10 大类 60+ 规则（含依赖安全分类）
  ├── prompt_builder.needs_multi_round(scan_result, skill_root) → 判断是否需要多轮分析
  │
  │   【单轮模式】（文件总量在 char_budget 内可装下）
  │   ├── prompt_builder.build_messages(scan_result, skill_root[, context_limit])
  │   │     → system prompt + user（预扫描结果 + 文件内容，按风险优先级排序，有字符预算）
  │   ├── llm_client.stream_chat(messages, max_tokens) → 调用大模型（流式或非流式）
  │   ├── main 捕获 ContextOverflowError → 用 N 作为 context_limit 重试（最多 3 次）
  │   └── SSE 流式返回：status / pre_scan / ai_chunk / done | error
  │
  │   【多轮模式】（文件总量超出单轮 char_budget）
  │   ├── prompt_builder.split_files_into_batches(scan_result, skill_root)
  │   │     → 将所有文件按 char_budget 分成 N 个批次（不按风险排序，防止对抗性规避）
  │   ├── 循环每个批次：
  │   │     ├── prompt_builder.build_batch_messages(scan_result, batch_files, idx, total)
  │   │     │     → BATCH_SYSTEM_PROMPT + 预扫描结果 + 该批次文件内容
  │   │     └── llm_client.stream_chat → 收集该批次的风险发现（JSON 格式，不流式输出给前端）
  │   ├── prompt_builder.build_merge_messages(scan_result, batch_results)
  │   │     → MERGE_SYSTEM_PROMPT + 预扫描结果 + 所有批次分析结果
  │   ├── llm_client.stream_chat → 生成最终汇总报告（流式输出给前端）
  │   └── SSE 流式返回：status / pre_scan / status(批次进度) / ai_chunk(最终报告) / done | error
```

- **前端**: `static/index.html` 单页，拖拽上传 ZIP，消费 SSE，用 marked 渲染 Markdown 报告，可下载 .md。  
- **配置单一来源**: 服务参数默认值在 `config.py`；LLM 模型配置已移至 Web UI 页面管理。

---

## 3. 核心模块说明

| 文件 | 职责 |
|------|------|
| `config.py` | 配置集中管理：LLM 参数通过 Web UI 动态配置；服务参数（上传限制、host/port）通过环境变量注入。 |
| `scanner.py` | 正则预扫描引擎。`PATTERNS` 定义 60+ 条规则（敏感信息含私钥/证书内容、危险操作含反向Shell/后门/SQL注入/XSS、数据泄露含系统信息收集、下载风险含系统持久化目录、供应链含代码托管平台链接、Prompt 安全、工具权限、依赖安全含恶意包名检测等）；`scan_file_structure` 含私钥文件名检测（.pem/.key/.p12/id_rsa等）；`run_full_scan(root)` 返回 structure（含 private_key_files）+ categories（含 dependency_sec）+ total_hits + file_hit_ranking + scanned_files。 |
| `prompt_builder.py` | 系统提示词（SYSTEM_PROMPT / BATCH_SYSTEM_PROMPT / MERGE_SYSTEM_PROMPT，对标 ciso SKILL.md 的 9 大维度与报告格式，含私钥/证书检测、反向Shell/后门、SQL注入/XSS、系统信息收集、系统持久化目录、代码托管平台链接、恶意包名检测等增强项）；`build_prescan_section`（含私钥文件输出）；`read_skill_files`（按 file_hit_ranking 优先，在 char_budget 内读入文件）；单轮：`build_messages` 返回 (messages, max_tokens)；多轮：`split_files_into_batches` 将文件分批（不按风险排序，防止对抗性规避）、`needs_multi_round` 判断是否需要多轮、`build_batch_messages` 构建单批次消息、`build_merge_messages` 构建汇总消息。token 估算用 `estimate_tokens(text) = len(text)//2`（保守，避免小上下文模型溢出）。 |
| `llm_client.py` | OpenAI 兼容 API：`stream_chat(messages, max_tokens)`，根据 `LLM_STREAM` 走流式或阻塞；`_blocking_mode` 在 400 时解析 "maximum context length is N" 并抛出 `ContextOverflowError(max_tokens=N, ...)`。 |
| `main.py` | FastAPI 应用：`/` 返回静态 index.html；`/health`、`/api/config`；`POST /api/audit` 接收 file，跑 `_audit_stream`（解压→预扫描→判断单轮/多轮→单轮：组装消息+调用LLM带context溢出重试；多轮：分批调用LLM收集各批次结果+汇总调用LLM生成最终报告），SSE 返回。 |

---

## 4. 配置与默认值（以 config.py 为准）

- **LLM 配置**：已移至 Web UI 页面动态管理（API 地址、模型、密钥、生成参数等），启动后在浏览器中配置。
- **服务参数**（通过环境变量）：`HOST=0.0.0.0`、`PORT=7860`、`MAX_UPLOAD_SIZE_MB=50`、`MAX_CONTENT_CHARS=500000`。

---

## 5. 大模型与上下文处理

- **模型**: 通过 Web UI 配置，支持任意 OpenAI 兼容 API。
- **流式**: 默认不流式，可在 Web UI 中开启。
- **单轮/多轮自动切换**: 系统自动判断文件总量是否在单轮 char_budget 内可装下：  
  - **单轮模式**: 文件量小，所有文件一次性送入大模型分析，支持 context 溢出自动重试（最多 3 次）。  
  - **多轮模式**: 文件量大，将所有文件按 char_budget 分成多个批次，每批独立调用大模型分析（输出 JSON 格式的风险发现），最后汇总所有批次结果调用大模型生成完整 Markdown 报告。  
- **防对抗性规避**: 多轮模式下文件分批**不按风险排序**，确保每个文件都被平等分析，防止恶意 Skill 通过操纵排序将高风险内容排到后面被截断。  
- **上下文溢出**: 单轮模式下若模型返回 400 且错误中含 "maximum context length is N"，则用 N 作为 `context_limit` 重新 `build_messages`，缩减文件内容与 max_tokens，最多重试 3 次。多轮模式下单批次溢出会跳过该批次并记录错误，汇总阶段溢出会报错。  
- **Token 估算**: 使用 `len(text)//2`，偏保守，以适配 8K 等小上下文模型。

---

## 6. Docker 与私有化部署约定

- **多架构**: 使用 `docker buildx`，`--platform linux/amd64` 或 `linux/arm64`。  
- **Tag 约定**: 单架构构建使用带架构后缀的 tag，避免同机 x86/ARM 互相覆盖：  
  - x86: `openclaw-skills-guard:latest-amd64`
  - ARM: `openclaw-skills-guard:latest-arm64`
  - 推送到仓库多架构时可用 `your-registry.com/openclaw-skills-guard:latest`。
- **离线**: 单架构用 `docker build` + `docker save/load`；多架构则分别 buildx 构建 amd64/arm64，再分别 save 为 `openclaw-skills-guard-amd64.tar` / `openclaw-skills-guard-arm64.tar`，在目标机按架构 load 后 run。  
- **Dockerfile**: 基于 `python:3.11-slim`，安装 curl（healthcheck 用），暴露 7860，HEALTHCHECK 调 `/health`。

---

## 7. 安全审计能力来源（ciso skill）

- 审计逻辑与分类以 `openclaw_skills/ciso/` 为参考：  
  - `SKILL.md`：检测维度、风险等级、报告结构、判断原则。  
  - `scripts/analyze.sh`：bash/grep 式规则 → 在 scanner.py 中实现为 Python 正则。  
  - `references/`：报告模板、检查清单等，用于设计 prompt_builder 的 SYSTEM_PROMPT 与输出格式。  
- **已完全对齐 ciso 能力**，并在以下方面做了增强：  
  - 私钥文件名检测（.pem/.key/.p12/.pfx/.jks/.keystore、id_rsa/id_dsa/id_ecdsa/id_ed25519）  
  - 私钥/证书内容检测（-----BEGIN PRIVATE KEY-----、-----BEGIN CERTIFICATE-----）  
  - 反向Shell/后门检测（reverse shell、backdoor、/dev/tcp/、nc -e、socat TCP）  
  - 不安全HTTP连接检测（非HTTPS外部连接）  
  - SQL注入/XSS风险检测  
  - 系统信息收集检测（os.uname、platform、hostname、psutil）  
  - 下载到系统/持久化目录检测（/usr/bin、.bashrc、systemd、crontab）  
  - 可疑代码托管平台链接检测（github releases、gitee releases、codeberg releases）  
  - 依赖安全正则检测（无版本锁定、已知恶意/typosquatting包名）  
- 报告输出为 Markdown，结构包含：基本信息、总体风险评级、高风险发现汇总、各维度详细分析、风险组合分析、使用建议、修复建议。

---

## 8. 测试与示例资源

- **测试用 Skill 压缩包**: `test_data_helper_skill.zip`（内为故意含有风险的 data-helper-skill：硬编码 API Key、AWS 示例密钥、eval、subprocess、requests.post、读取 ~/.ssh/~/.aws、curl|bash、社会工程学文案等），用于验证预扫描与大模型报告。  
- **同目录下** 可能还有解压后的 `data-helper-skill/` 目录（与 zip 内容一致），仅作参考，部署与运行不依赖该目录。

---

## 9. 已做过的修正

- **README 多架构构建**: 单架构用带架构后缀的 tag（`openclaw-skills-guard:latest-amd64`/`latest-arm64`），避免覆盖。
- **离线部署**: 拆分为”单架构”和”多架构”两种方式。
- **LLM 配置迁移至 Web UI**: Dockerfile 和 .env.example 中已移除所有 LLM 硬编码默认值，启动后通过 Web UI 配置。

---

## 10. 项目文件树（主要文件）

```
openclaw_skills_guard/
├── config.py           # 配置
├── main.py             # FastAPI 入口与 /api/audit 管道
├── scanner.py          # 正则预扫描
├── prompt_builder.py   # 提示词与消息组装
├── llm_client.py       # 大模型调用与上下文溢出处理
├── requirements.txt
├── .env.example
├── Dockerfile
├── .dockerignore
├── README.md
├── build_context.md    # 本文件
├── static/
│   └── index.html      # 上传页与报告展示
├── test_data_helper_skill.zip   # 测试用“有风险”Skill 包
└── data-helper-skill/  # 可选，与 zip 内容一致
```

---

## 11. 常用命令速查

```bash
# 本地跑
pip install -r requirements.txt && python main.py   # 默认 http://0.0.0.0:7860

# Docker 单架构
docker build -t openclaw-skills-guard:latest .
docker run -d -p 7860:7860 --name openclaw-skills-guard openclaw-skills-guard:latest

# Docker 多架构（buildx）
docker buildx build --platform linux/amd64 -t openclaw-skills-guard:latest-amd64 --load .
docker buildx build --platform linux/arm64 -t openclaw-skills-guard:latest-arm64 --load .
```

以上为 AicoGuardClaw 的完整构建与运行上下文，后续改动（如新增检测规则、改报告格式、换模型默认值）应保持与本文档及 config.py 一致。
