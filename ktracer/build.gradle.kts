plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
    application
}

group = "net.skai.ktracer"
version = "1.0-SNAPSHOT"

repositories {
    // sk-ai-net/skainet's GitHub Packages repo (needing GITHUB_USER/GITHUB_TOKEN) is gone —
    // the project publishes to Maven Central as sk.ainet.core:* now.
    mavenCentral()
}

dependencies {
    implementation(libs.skainet.io.gguf)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation(libs.kotlinx.io.core)

    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    // skainet-io-gguf-jvm:0.40.1 is compiled targeting Java 21 class files (version 65) —
    // 17 fails with UnsupportedClassVersionError at runtime.
    jvmToolchain(21)
}

application {
    mainClass.set("net.skai.ktracer.MainKt")
}
