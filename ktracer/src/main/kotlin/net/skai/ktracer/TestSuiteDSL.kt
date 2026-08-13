package net.skai.ktracer

import java.nio.file.Path
import kotlin.io.path.*

/**
 * Test case definition that holds inputs and expected result
 */
class TestCase(
    val description: String,
    var inputs: List<Any> = emptyList(),
    var result: Any? = null
) {
    internal fun validate() {
        require(result != null) { "Test case must have a result" }
    }
}

/**
 * Test suite that contains multiple test cases
 */
class TestSuite(val name: String) {
    private val cases = mutableListOf<TestCase>()
    
    fun test(description: String, block: TestCase.() -> Unit) {
        val testCase = TestCase(description)
        testCase.apply(block)
        testCase.validate()
        cases.add(testCase)
    }
    
    internal fun getCases() = cases.toList()
}

/**
 * Use case that contains multiple test suites
 */
class UseCase(val name: String) {
    private val suites = mutableListOf<TestSuite>()
    
    fun suite(name: String, block: TestSuite.() -> Unit) {
        val testSuite = TestSuite(name)
        testSuite.apply(block)
        suites.add(testSuite)
    }
    
    internal fun getSuites() = suites.toList()
}

/**
 * Test runner that executes test cases and stores results
 */
class TestRunner(private val outputPath: Path) {
    fun useCase(name: String, block: UseCase.() -> Unit): UseCase {
        val useCase = UseCase(name)
        useCase.apply(block)
        return useCase
    }
    
    fun execute(useCase: UseCase) {
        useCase.getSuites().forEach { suite ->
            val suitePath = outputPath.resolve("TS-${suite.name}")
            suitePath.createDirectories()

            suite.getCases().forEach { case ->
                val ggufPath = suitePath.resolve("UC-${useCase.name}_${case.hashCode()}.gguf")

                // Store test case in GGUF format
                val tensors = mutableMapOf<String, Any>()
                case.inputs.forEachIndexed { index, input ->
                    tensors["input_$index"] = input
                }
                case.result?.let { tensors["result"] = it }

                writeGgufTestData(ggufPath, mapOf("experiment_description" to case.description), tensors)
            }
        }
    }
}

// Extension functions for building test cases
fun TestCase.input(vararg values: Any) {
    inputs = values.toList()
}

fun TestCase.expect(value: Any) {
    result = value
}

// Example usage:
/*
val runner = TestRunner(Path("/path/to/output"))

runner.useCase("matrix_ops") {
    suite("1") {
        test("Matrix multiplication") {
            input(
                createTensor(2, 2) { 1f },
                createTensor(2, 2) { 2f }
            )
            expect(createTensor(2, 2) { 4f })
        }
        
        test("Matrix addition") {
            input(
                createTensor(2, 2) { 1f },
                createTensor(2, 2) { 2f }
            )
            expect(createTensor(2, 2) { 3f })
        }
    }
    
    suite("2") {
        test("Matrix transpose") {
            input(createTensor(2, 3) { it.toFloat() })
            expect(createTensor(3, 2) { it.toFloat() })
        }
    }
}.also { runner.execute(it) }
*/
