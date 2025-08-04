package com.amb.aivision

data class Model(
    val name: String,
    val url: String,
    val version: String = "1.0",
    val totalBytes: Long,
    val downloadFileName: String,
    val normalizedName: String,
    val isZip: Boolean = false,
    val unzipDir: String? = null,
    val extraDataFiles: List<ExtraDataFile> = emptyList(),
    val accessToken: String? = null
)

data class ExtraDataFile(
    val url: String,
    val downloadFileName: String,
    val sizeInBytes: Long
)

data class ModelDownloadStatus(
    val status: ModelDownloadStatusType,
    val totalBytes: Long = 0,
    val receivedBytes: Long = 0,
    val bytesPerSecond: Long = 0,
    val remainingMs: Long = 0,
    val errorMessage: String = ""
)

enum class ModelDownloadStatusType {
    IN_PROGRESS, UNZIPPING, SUCCEEDED, FAILED, NOT_DOWNLOADED
}