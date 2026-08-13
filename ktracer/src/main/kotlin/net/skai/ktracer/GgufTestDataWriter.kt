package net.skai.ktracer

import java.nio.file.Path

/**
 * Shared by [TestRunner.execute] (TestSuiteDSL.kt) and [TestSuiteExecutor.storeResult] — both
 * referenced a `GGUFWriter(path).use { writer.addMetadata(...); writer.addTensor(...) }` shape
 * that never resolved against any dependency this module has actually had (pre-existing,
 * unrelated to the sk.ai.net:gguf -> sk.ainet.io.gguf migration).
 *
 * The current library's writer — `sk.ainet.io.gguf.export.GGUFWriter.writeToByteArray`/
 * `writeToSink` — takes a `GgufWriteRequest(metadata: Map<String, Any>, tensors:
 * List<GgufTensorEntry>, tensorMap: Map<String, String>)`, not a mutable builder. Porting this
 * needs a real mapping from the ad-hoc `tensors: Map<String, Any>` callers build here (arbitrary
 * values — could be a FloatArray, a List<Float>, anything a test case's `input(...)`/`expect(...)`
 * was given) onto `GgufTensorEntry`'s typed shape (GGML quantization type + dimensions per
 * tensor) — a design decision for whoever owns this DSL, not something to infer from `Any`.
 */
internal fun writeGgufTestData(ggufPath: Path, metadata: Map<String, String>, tensors: Map<String, Any>): Nothing =
    TODO("Port to sk.ainet.io.gguf.export.GGUFWriter's GgufWriteRequest/GgufTensorEntry API")
