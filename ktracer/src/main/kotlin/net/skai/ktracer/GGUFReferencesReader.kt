package net.skai.ktracer

import kotlinx.io.asSource
import kotlinx.io.buffered
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.io.gguf.GGUFValueType
import java.io.File
import java.nio.file.Path
import kotlin.io.path.Path
import kotlin.io.path.inputStream

/**
 * Utility class for reading GGUF files produced by the Python gradienttracer
 */
class GGUFReferencesReader(private val basePath: Path) {

    /**
     * Reads a GGUF file and returns its contents
     * @param filePath Path to the GGUF file
     * @return Map of tensor names to their values
     */
    fun readGGUF(filePath: Path): Map<String, Any> {
        val reader = GGUFReader(filePath.inputStream().asSource().buffered())
        return reader.tensors.associate { it.name to it.data }
    }

    /**
     * Iterates over all GGUF files in a directory recursively
     * @return Sequence of pairs of file paths and their tensor contents
     */
    fun iterateGGUFFiles(): Sequence<Pair<Path, Map<String, Any>>> = sequence {
        File(basePath.toUri()).walk()
            .filter { it.isFile && it.extension == "gguf" }
            .forEach { file ->
                val path = Path(file.absolutePath)
                yield(path to readGGUF(path))
            }
    }


    /** Retrieve a metadata field as a String, if it exists and is of type string */
    fun GGUFReader.getString(key: String): String {
        val field = this.fields[key] ?: return ""  // key not present
        // Ensure it's a singular string (not an array)
        if (field.types.size == 1 && field.types[0] == GGUFValueType.STRING) {
            // The actual string bytes are at the index specified by data[0]
            val byteList = field.parts[field.data[0]] as? List<UByte> ?: return ""
            return byteList.toUByteArray().toByteArray().decodeToString()  // UTF-8 decoding
        }
        return ""  // Not a single string field
    }

    /** Retrieve a metadata field as a list of Strings (for array-of-string fields) */
    fun GGUFReader.getStringList(key: String): List<String> {
        val field = this.fields[key] ?: return emptyList()
        // Expect an array of strings: types[0] == ARRAY and types[1] == STRING (per format)
        if (field.types.size >= 2 &&
            field.types[0] == GGUFValueType.ARRAY && field.types[1] == GGUFValueType.STRING
        ) {
            // Each entry in data corresponds to one string's bytes in parts
            return field.data.map { idx ->
                val byteList = field.parts[idx] as List<UByte>
                byteList.toUByteArray().toByteArray().decodeToString()
            }
        }
        return emptyList()  // Not an array-of-strings field
    }


    /**
     * Gets experiment description from a GGUF file
     * @param filePath Path to the GGUF file
     * @return Experiment description string
     *
     */
    fun getExperimentDescription(filePath: Path): String {
        val reader = GGUFReader(filePath.inputStream().asSource().buffered())
        return reader.getString("experiment_description")
    }
}
