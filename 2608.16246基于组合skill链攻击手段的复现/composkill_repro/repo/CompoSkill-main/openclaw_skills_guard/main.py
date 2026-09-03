"""
AicoGuardClaw — AI Skill Security Audit Service
FastAPI application entry point.
"""

import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import llm_client
import prompt_builder
import scanner
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("openclaw_skills_guard")

app = FastAPI(title="AicoGuardClaw", description="AI Skill 安全审计系统")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    return {"status": "healthy", "model": settings.llm_model}


def _sse_event(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"


async def _summarize_batch_results(batch_results, context_limit, total_batches):
    """
    If batch results exceed the merge context budget, use LLM to summarize
    each oversized batch result. If LLM summarization still exceeds budget,
    falls back to hard truncation as a last resort.
    Returns (compressed_batch_results, sse_events).
    """
    available_budget, needs_summarize = prompt_builder.compute_merge_char_budget(
        batch_results, context_limit
    )
    if not needs_summarize:
        return batch_results, []

    sse_events = []
    total_chars = sum(len(r) for r in batch_results)
    sse_events.append(_sse_event("status", {
        "message": f"📝 批次结果总量({total_chars}字符)超出汇总预算({available_budget}字符)，"
                   f"正在用大模型对 {total_batches} 个批次结果进行摘要压缩..."
    }))

    # Calculate per-batch target: proportionally allocate budget
    summarized_results = []

    for idx, result in enumerate(batch_results, 1):
        proportion = len(result) / total_chars if total_chars > 0 else 1.0 / len(batch_results)
        target_chars = max(200, int(available_budget * proportion))

        if len(result) <= target_chars:
            summarized_results.append(result)
            continue

        sse_events.append(_sse_event("status", {
            "message": f"📝 正在摘要压缩第 {idx}/{total_batches} 批结果（{len(result)} → ~{target_chars} 字符）..."
        }))

        summarize_messages, summarize_max_tokens = prompt_builder.build_summarize_messages(
            result, target_chars=target_chars, context_limit=context_limit
        )
        logger.info(
            "Summarizing batch %d/%d: input=%d chars, target=%d chars, max_output_tokens=%d",
            idx, total_batches, len(result), target_chars, summarize_max_tokens,
        )

        summary_chunks = []
        try:
            async for chunk in llm_client.stream_chat(
                summarize_messages, max_tokens=summarize_max_tokens
            ):
                summary_chunks.append(chunk)
        except llm_client.ContextOverflowError as summarize_err:
            logger.warning(
                "Summarize batch %d overflow (max=%d), hard-truncating to %d chars",
                idx, summarize_err.max_tokens, target_chars,
            )
            summarized_results.append(result[:target_chars] + "\n...(因上下文限制截断)")
            continue

        summary_text = "".join(summary_chunks)

        # If LLM output still exceeds target, hard-truncate as last resort
        if len(summary_text) > target_chars:
            logger.warning(
                "Batch %d/%d summary still too long (%d > %d), hard-truncating",
                idx, total_batches, len(summary_text), target_chars,
            )
            summary_text = summary_text[:target_chars] + "\n...(摘要超长截断)"

        logger.info(
            "Batch %d/%d summarized: %d → %d chars",
            idx, total_batches, len(result), len(summary_text),
        )
        summarized_results.append(summary_text)

    # Final safety check: if total still exceeds budget, proportionally truncate all
    final_total = sum(len(r) for r in summarized_results)
    if final_total > available_budget:
        logger.warning(
            "Post-summarize total (%d) still exceeds budget (%d), applying final truncation",
            final_total, available_budget,
        )
        sse_events.append(_sse_event("status", {
            "message": f"⚠️ 摘要后总量({final_total}字符)仍超预算({available_budget}字符)，进行最终截断..."
        }))
        final_results = []
        for result in summarized_results:
            proportion = len(result) / final_total if final_total > 0 else 1.0 / len(summarized_results)
            budget = max(150, int(available_budget * proportion))
            if len(result) > budget:
                final_results.append(result[:budget] + "\n...(最终截断)")
            else:
                final_results.append(result)
        summarized_results = final_results

    return summarized_results, sse_events


async def _audit_stream(file_content: bytes):
    """Core audit pipeline — yields SSE events."""
    tmp_dir = tempfile.mkdtemp(prefix="aicoguardclaw_")
    extract_dir = os.path.join(tmp_dir, "skill")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        yield _sse_event("status", {"message": "📦 正在接收并解压文件..."})

        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            yield _sse_event("error", {
                "message": f"文件大小 ({size_mb:.1f}MB) 超过限制 ({settings.max_upload_size_mb}MB)"
            })
            return

        zip_path = os.path.join(tmp_dir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(file_content)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            yield _sse_event("error", {"message": "无效的ZIP文件，请上传标准ZIP压缩包"})
            return

        # if zip contains a single top-level directory, use that as root
        entries = os.listdir(extract_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
            skill_root = os.path.join(extract_dir, entries[0])
        else:
            skill_root = extract_dir

        # --- Phase 1: regex pre-scan ---
        yield _sse_event("status", {"message": "🔍 正在进行正则预扫描..."})
        scan_result = scanner.run_full_scan(skill_root)
        yield _sse_event("pre_scan", {"data": scan_result})

        risk_summary = f"共扫描 {scan_result['structure']['total_files']} 个文件，" \
                       f"正则命中 {scan_result['total_hits']} 处"
        yield _sse_event("status", {"message": f"🔍 预扫描完成：{risk_summary}"})

        # --- Phase 2: LLM analysis ---
        context_limit = 0  # 0 = use default from settings
        multi_round = prompt_builder.needs_multi_round(
            scan_result, skill_root, context_limit
        )

        if multi_round:
            # --- Phase 2 (multi-round): batch analysis + merge ---
            batches = prompt_builder.split_files_into_batches(
                scan_result, skill_root, context_limit
            )
            total_batches = len(batches)
            yield _sse_event("status", {
                "message": f"🤖 文件量较大，将分 {total_batches} 批调用大模型 ({settings.llm_model}) 进行深度分析..."
            })

            batch_results = []
            for batch_idx, batch_files in enumerate(batches, 1):
                batch_file_names = [rel for rel, _ in batch_files]
                yield _sse_event("status", {
                    "message": f"🔄 正在分析第 {batch_idx}/{total_batches} 批（{len(batch_file_names)} 个文件）..."
                })

                batch_messages, batch_max_tokens = prompt_builder.build_batch_messages(
                    scan_result, batch_files, batch_idx, total_batches,
                    context_limit=context_limit,
                )
                input_est = prompt_builder.estimate_tokens(
                    batch_messages[0]["content"] + batch_messages[1]["content"]
                )
                logger.info(
                    "Batch %d/%d: ~%d input tokens, max_output=%d, files=%s",
                    batch_idx, total_batches, input_est, batch_max_tokens,
                    batch_file_names,
                )

                batch_chunks = []
                try:
                    async for chunk in llm_client.stream_chat(
                        batch_messages, max_tokens=batch_max_tokens
                    ):
                        batch_chunks.append(chunk)
                except llm_client.ContextOverflowError as overflow_err:
                    logger.warning(
                        "Batch %d context overflow (model max=%d), skipping",
                        batch_idx, overflow_err.max_tokens,
                    )
                    batch_chunks = [
                        json.dumps({"error": f"批次{batch_idx}因上下文溢出被跳过",
                                    "files": batch_file_names}, ensure_ascii=False)
                    ]

                batch_result_text = "".join(batch_chunks)
                batch_results.append(batch_result_text)
                logger.info("Batch %d/%d completed, result length=%d chars",
                            batch_idx, total_batches, len(batch_result_text))

            # --- Phase 2c: summarize if needed, then merge ---
            batch_results, summarize_events = await _summarize_batch_results(
                batch_results, context_limit, total_batches
            )
            for event in summarize_events:
                yield event

            yield _sse_event("status", {
                "message": f"📊 所有 {total_batches} 批分析完成，正在汇总生成最终报告..."
            })

            merge_messages, merge_max_tokens = prompt_builder.build_merge_messages(
                scan_result, batch_results, context_limit=context_limit,
            )
            merge_input_est = prompt_builder.estimate_tokens(
                merge_messages[0]["content"] + merge_messages[1]["content"]
            )
            logger.info(
                "Merge round: ~%d input tokens, max_output=%d",
                merge_input_est, merge_max_tokens,
            )

            report_chunks = []
            try:
                async for chunk in llm_client.stream_chat(
                    merge_messages, max_tokens=merge_max_tokens
                ):
                    report_chunks.append(chunk)
                    yield _sse_event("ai_chunk", {"content": chunk})
            except llm_client.ContextOverflowError as overflow_err:
                logger.error("Merge round context overflow: %s", overflow_err)
                yield _sse_event("error", {
                    "message": f"汇总阶段超出模型上下文限制({overflow_err.max_tokens}tokens)"
                })

        else:
            # --- Phase 2 (single-round): try single call, fallback to multi-round ---
            yield _sse_event("status", {
                "message": f"🤖 正在调用大模型 ({settings.llm_model}) 进行深度分析..."
            })

            messages, max_tokens = prompt_builder.build_messages(
                scan_result, skill_root, context_limit=context_limit
            )
            input_est = prompt_builder.estimate_tokens(
                messages[0]["content"] + messages[1]["content"]
            )
            logger.info(
                "LLM single-round: ~%d input tokens, max_output=%d, context_limit=%s",
                input_est, max_tokens, context_limit or "auto",
            )

            try:
                report_chunks = []
                async for chunk in llm_client.stream_chat(
                    messages, max_tokens=max_tokens
                ):
                    report_chunks.append(chunk)
                    yield _sse_event("ai_chunk", {"content": chunk})
            except llm_client.ContextOverflowError as overflow_err:
                # Single-round overflow — fallback to multi-round batch mode
                # to ensure NO content is discarded.
                logger.warning(
                    "Single-round context overflow (model max=%d), "
                    "falling back to multi-round batch mode",
                    overflow_err.max_tokens,
                )
                context_limit = overflow_err.max_tokens
                report_chunks = []

                batches = prompt_builder.split_files_into_batches(
                    scan_result, skill_root, context_limit
                )
                total_batches = len(batches)
                yield _sse_event("status", {
                    "message": f"⚠️ 模型上下文限制为{overflow_err.max_tokens}tokens，"
                               f"自动切换为分批模式（共 {total_batches} 批），确保所有文件完整分析..."
                })

                batch_results = []
                for batch_idx, batch_files in enumerate(batches, 1):
                    batch_file_names = [rel for rel, _ in batch_files]
                    yield _sse_event("status", {
                        "message": f"🔄 正在分析第 {batch_idx}/{total_batches} 批（{len(batch_file_names)} 个文件）..."
                    })

                    batch_messages, batch_max_tokens = prompt_builder.build_batch_messages(
                        scan_result, batch_files, batch_idx, total_batches,
                        context_limit=context_limit,
                    )
                    batch_input_est = prompt_builder.estimate_tokens(
                        batch_messages[0]["content"] + batch_messages[1]["content"]
                    )
                    logger.info(
                        "Fallback batch %d/%d: ~%d input tokens, max_output=%d, files=%s",
                        batch_idx, total_batches, batch_input_est, batch_max_tokens,
                        batch_file_names,
                    )

                    batch_chunks = []
                    try:
                        async for chunk in llm_client.stream_chat(
                            batch_messages, max_tokens=batch_max_tokens
                        ):
                            batch_chunks.append(chunk)
                    except llm_client.ContextOverflowError as batch_overflow:
                        logger.warning(
                            "Fallback batch %d context overflow (model max=%d), skipping",
                            batch_idx, batch_overflow.max_tokens,
                        )
                        batch_chunks = [
                            json.dumps({"error": f"批次{batch_idx}因上下文溢出被跳过",
                                        "files": batch_file_names}, ensure_ascii=False)
                        ]

                    batch_result_text = "".join(batch_chunks)
                    batch_results.append(batch_result_text)
                    logger.info("Fallback batch %d/%d completed, result length=%d chars",
                                batch_idx, total_batches, len(batch_result_text))

                # Summarize if needed, then merge all batch results
                batch_results, summarize_events = await _summarize_batch_results(
                    batch_results, context_limit, total_batches
                )
                for event in summarize_events:
                    yield event

                yield _sse_event("status", {
                    "message": f"📊 所有 {total_batches} 批分析完成，正在汇总生成最终报告..."
                })

                merge_messages, merge_max_tokens = prompt_builder.build_merge_messages(
                    scan_result, batch_results, context_limit=context_limit,
                )
                merge_input_est = prompt_builder.estimate_tokens(
                    merge_messages[0]["content"] + merge_messages[1]["content"]
                )
                logger.info(
                    "Fallback merge round: ~%d input tokens, max_output=%d",
                    merge_input_est, merge_max_tokens,
                )

                try:
                    async for chunk in llm_client.stream_chat(
                        merge_messages, max_tokens=merge_max_tokens
                    ):
                        report_chunks.append(chunk)
                        yield _sse_event("ai_chunk", {"content": chunk})
                except llm_client.ContextOverflowError as merge_overflow:
                    logger.error("Fallback merge round context overflow: %s", merge_overflow)
                    yield _sse_event("error", {
                        "message": f"汇总阶段超出模型上下文限制({merge_overflow.max_tokens}tokens)"
                    })

        full_report = "".join(report_chunks)
        yield _sse_event("done", {
            "message": "✅ 审计完成",
            "report_length": len(full_report),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as exc:
        logger.exception("Audit pipeline error")
        yield _sse_event("error", {"message": f"审计过程异常: {exc}"})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/audit")
async def audit(file: UploadFile = File(...)):
    """Upload a ZIP of skill files → stream back audit results via SSE."""
    if not settings.model_connected:
        return StreamingResponse(
            iter([_sse_event("error", {"message": "请先完成模型配置并通过联通性测试"})]),
            media_type="text/event-stream",
        )

    if not file.filename or not file.filename.lower().endswith(".zip"):
        return StreamingResponse(
            iter([_sse_event("error", {"message": "请上传 .zip 格式的压缩包"})]),
            media_type="text/event-stream",
        )

    file_content = await file.read()

    return StreamingResponse(
        _audit_stream(file_content),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/config")
async def get_config():
    """Return all user-configurable LLM parameters + connection status."""
    return settings.get_llm_config_dict()


@app.post("/api/test-connection")
async def test_connection(body: dict = None):
    """Save config + test LLM connectivity in one step. Updates model_connected status."""
    if body:
        settings.update_llm_config(
            llm_api_url=body.get("llm_api_url"),
            llm_model=body.get("llm_model"),
            llm_api_key=body.get("llm_api_key"),
            max_context_tokens=body.get("max_context_tokens"),
            max_output_tokens=body.get("max_output_tokens"),
        )
        logger.info("LLM config updated via test-connection: model=%s, url=%s", settings.llm_model, settings.llm_api_url)
    result = await llm_client.test_connection(timeout=30.0)
    settings.model_connected = result["success"]
    logger.info("Connection test result: success=%s, message=%s", result["success"], result["message"])
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
