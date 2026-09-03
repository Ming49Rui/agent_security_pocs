# AicoGuardClaw — AI Skill 安全审计系统

基于 **正则预扫描 + 大模型语义分析** 双引擎的 AI Skill 安全审计服务。

用户上传 Skill 文件压缩包 → 服务端自动扫描 + 调用大模型分析 → 输出结构化安全报告。

## 架构

```
用户浏览器（上传ZIP）
        ↓ HTTP
FastAPI 服务端
  ├── 正则预扫描引擎（scanner.py, 9大类 50+规则）
  ├── 提示词组装（prompt_builder.py）
  └── 大模型API调用（llm_client.py, OpenAI兼容格式）
        ↓
大模型（qwen/claude/gpt 等）
        ↓
SSE 流式返回 → 前端实时渲染 Markdown 报告
```

## 检测能力（9大维度）

| # | 维度 | 说明 |
|---|------|------|
| 1 | 文件结构分析 | 类型分布、隐藏文件、大文件 |
| 2 | 敏感信息检测 | API Key、云AK/SK、JWT、私钥 |
| 3 | 危险操作检测 | 系统命令、文件写入、Shell危险命令 |
| 4 | 数据泄露风险 | 网络上传+数据收集组合、敏感文件外发 |
| 5 | 外部下载风险 | 下载执行、管道执行、权限修改 |
| 6 | 供应链风险 | 可疑链接、安装命令引导、社会工程学 |
| 7 | Prompt安全 | 注入、泄漏、隐藏指令、越狱 |
| 8 | 工具权限滥用 | allowed-tools过度声明、通配符权限 |
| 9 | 依赖安全 | 外部依赖可信度、已知漏洞 |

## 快速开始

### 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置（可选）
cp .env.example .env
# 编辑 .env 修改服务参数；LLM 模型配置在启动后通过 Web UI 进行

# 3. 启动服务
python main.py

# 4. 打开浏览器
# http://localhost:7860
```

### Docker 运行

```bash
# 构建镜像
docker build -t openclaw-skills-guard:latest .

# 运行（使用默认配置）
docker run -d -p 7860:7860 --name openclaw-skills-guard openclaw-skills-guard:latest

# 查看日志
docker logs -f openclaw-skills-guard
# LLM 模型配置在启动后访问 http://localhost:7860 通过 Web UI 进行
```

## 私有化部署

### 多架构构建（x86 + ARM）

```bash
# 创建 buildx 构建器（首次需要）
docker buildx create --name multi-builder --use

# 构建并推送到私有仓库（同时支持 x86 和 ARM）
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t your-registry.com/openclaw-skills-guard:latest \
  --push .

# 仅构建 x86（tag 带架构后缀，避免与 ARM 镜像覆盖）
docker buildx build --platform linux/amd64 -t openclaw-skills-guard:latest-amd64 --load .

# 仅构建 ARM
docker buildx build --platform linux/arm64 -t openclaw-skills-guard:latest-arm64 --load .
```

### 单架构构建（当前机器架构）

```bash
docker build -t openclaw-skills-guard:latest .
```

### 离线部署

**方式一：单架构（当前机器架构）**

```bash
# 1. 在有网络的机器上构建并导出
docker build -t openclaw-skills-guard:latest .
docker save openclaw-skills-guard:latest -o openclaw-skills-guard.tar

# 2. 传输到目标机器后导入
docker load -i openclaw-skills-guard.tar

# 3. 启动（LLM 配置在 Web UI 中进行）
docker run -d -p 7860:7860 --name openclaw-skills-guard openclaw-skills-guard:latest
```

**方式二：多架构（x86 与 ARM 分别导出）**

```bash
# 1. 创建 buildx 构建器（首次需要）
docker buildx create --name multi-builder --use

# 2. 分别构建并导出
docker buildx build --platform linux/amd64 -t openclaw-skills-guard:latest-amd64 --load .
docker save openclaw-skills-guard:latest-amd64 -o openclaw-skills-guard-amd64.tar

docker buildx build --platform linux/arm64 -t openclaw-skills-guard:latest-arm64 --load .
docker save openclaw-skills-guard:latest-arm64 -o openclaw-skills-guard-arm64.tar

# 3. 将对应架构的 tar 传输到目标机器后导入并启动
# x86 目标机：
docker load -i openclaw-skills-guard-amd64.tar
docker run -d -p 7860:7860 --name openclaw-skills-guard openclaw-skills-guard:latest-amd64

# ARM 目标机：
docker load -i openclaw-skills-guard-arm64.tar
docker run -d -p 7860:7860 --name openclaw-skills-guard openclaw-skills-guard:latest-arm64
```

### Docker Compose

```yaml
version: '3.8'
services:
  skills-guard:
    build: .
    ports:
      - "7860:7860"
    environment:
      - MAX_UPLOAD_SIZE_MB=50
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860/health"]
      interval: 30s
      timeout: 5s
      retries: 3
```

## 环境变量

> **注意**：LLM 模型配置（API 地址、模型名称、密钥、生成参数等）已移至 Web UI 页面，启动服务后在浏览器中配置即可。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_UPLOAD_SIZE_MB` | `50` | 上传文件大小限制 |
| `MAX_CONTENT_CHARS` | `500000` | 文件内容最大字符数 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `7860` | 监听端口 |

## 使用方法

1. 将待检测的 Skill 文件夹打包为 ZIP
2. 打开浏览器访问 `http://host:7860`
3. 拖拽或点击上传 ZIP 文件
4. 等待预扫描结果 + AI 分析报告
5. 可下载 Markdown 格式报告

## 技术栈

- **后端**: Python 3.11 / FastAPI / uvicorn
- **预扫描**: 纯 Python 正则引擎（50+ 安全规则）
- **大模型**: OpenAI 兼容 API（支持 qwen/claude/gpt 等）
- **前端**: 单页 HTML（原生 JS，marked.js 渲染 Markdown）
- **部署**: Docker 多架构（x86/ARM）
