package net.skai.ktracer

import kotlinx.coroutines.*
import kotlinx.io.asSource
import kotlinx.io.buffered
import java.nio.file.Path
import kotlin.io.path.*

/**
 * Data class representing a test execution result
 */
data class TestExecutionResult(
    val name: String,
    val description: String,
    val inputs: List<Any>,
    val result: Any,
    val testSuite: String,
    val useCase: String,
    val executionTimeMs: Long,
    val success: Boolean,
    val error: String? = null
)

/**
 * Class responsible for executing tests and storing results
 */
class TestSuiteExecutor(
    private val outputPath: Path,
    private val generateDot: Boolean = false,
    private val parallelExecution: Boolean = true
) {
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    /**
     * Execute a test case and capture its result
     */
    private suspend fun executeTest(
        testCase: TestCase,
        suiteName: String,
        useCaseName: String
    ): TestExecutionResult = withContext(Dispatchers.Default) {
        try {
            val startTime = System.currentTimeMillis()

            // TODO: TestCase (TestSuiteDSL.kt) has no execute() — the DSL's expect(...) sets
            // `result` directly, there's no notion of computing it at runtime. This function was
            // already unresolvable before this migration (pre-existing, unrelated to the
            // dependency swap); left as a stub since inventing execution semantics isn't a safe
            // guess. This whole class is dead code — nothing in Main.kt calls it.
            val result = testCase.result ?: error("TestCase '${testCase.description}' has no result to store")

            val executionTime = System.currentTimeMillis() - startTime
            
            TestExecutionResult(
                name = "${suiteName}_${useCaseName}_${testCase.description.replace(" ", "_")}",
                description = testCase.description,
                inputs = testCase.inputs,
                result = result,
                testSuite = suiteName,
                useCase = useCaseName,
                executionTimeMs = executionTime,
                success = true
            )
        } catch (e: Exception) {
            TestExecutionResult(
                name = "${suiteName}_${useCaseName}_${testCase.description.replace(" ", "_")}",
                description = testCase.description,
                inputs = testCase.inputs,
                result = Any(),
                testSuite = suiteName,
                useCase = useCaseName,
                executionTimeMs = 0,
                success = false,
                error = e.message
            )
        }
    }

    /**
     * Store test result in GGUF format
     */
    private fun storeResult(result: TestExecutionResult) {
        val tsFolder = outputPath.resolve("TS-${result.testSuite}")
        tsFolder.createDirectories()
        
        val ggufPath = tsFolder.resolve("${result.name}.gguf")

        val metadata = buildMap {
            put("experiment_description", result.description)
            put("execution_time_ms", result.executionTimeMs.toString())
            put("success", result.success.toString())
            result.error?.let { put("error", it) }
        }
        val tensors = buildMap {
            result.inputs.forEachIndexed { index, input -> put("input_$index", input) }
            put("result", result.result)
        }
        writeGgufTestData(ggufPath, metadata, tensors)

        // TODO: trace()/dag_2_dot() don't exist anywhere in this codebase either — this looks
        // like a port of gt/dot's Python DAG-visualization concept (gt/dot/dag.py, dag2dot.py)
        // that was never actually written on the Kotlin side. Pre-existing, unrelated to this
        // migration. Left disabled rather than stubbed with fabricated graph-tracing logic.
        if (generateDot && result.success) {
            error("DOT visualization (trace/dag_2_dot) isn't implemented on the Kotlin side yet")
        }
    }

    /**
     * Execute all test cases in a use case
     */
    suspend fun executeUseCase(useCase: UseCase): List<TestExecutionResult> = coroutineScope {
        val results = mutableListOf<TestExecutionResult>()
        
        useCase.getSuites().forEach { suite ->
            val testCases = suite.getCases()
            
            if (parallelExecution) {
                // Execute test cases in parallel
                val deferredResults = testCases.map { testCase ->
                    async {
                        executeTest(testCase, suite.name, useCase.name)
                    }
                }
                
                deferredResults.awaitAll().forEach { result ->
                    results.add(result)
                    storeResult(result)
                }
            } else {
                // Execute test cases sequentially
                testCases.forEach { testCase ->
                    val result = executeTest(testCase, suite.name, useCase.name)
                    results.add(result)
                    storeResult(result)
                }
            }
        }
        
        results
    }

    /**
     * Execute a use case and return results
     */
    fun execute(useCase: UseCase): List<TestExecutionResult> = runBlocking {
        executeUseCase(useCase)
    }

    /**
     * Clean up resources
     */
    fun close() {
        scope.cancel()
    }
}
