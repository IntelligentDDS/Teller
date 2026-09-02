/*
 * TELLER CUPTI injection: buffered write, output under TELLER_TRACE_DIR or /data/trace.
 * Short keys for high-frequency events (KERNEL, RUNTIME, DRIVER) to reduce I/O.
 * Compatible with CUDA 11+ (CUpti_ActivityKernel9); for older CUDA adjust kernel struct.
 */
#include <cupti.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <mutex>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>

static bool tracingEnabled = false;

static const std::vector<std::string> allowed_prefixes = {
    "cudaLaunchKernel", "cudaGraphLaunch", "cudaMemcpy", "cudaMemset",
    "cudaMalloc", "cudaFree", "cudaIpc", "cudaHostAlloc",
    "cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaStreamWaitEvent"
};

static bool has_allowed_prefix(const char* funcName) {
    if (!funcName) return false;
    for (const auto& prefix : allowed_prefixes) {
        if (strncmp(funcName, prefix.c_str(), prefix.size()) == 0) return true;
    }
    return false;
}

// ----- Buffered write (single-stage, lower overhead) -----
static const size_t WRITE_BUF_SIZE = 256 * 1024;  // 256KB
static std::vector<char> g_write_buf;
static size_t g_write_len = 0;
static FILE* g_fp = nullptr;
static std::mutex g_mutex;
static bool g_buf_initialized = false;

static void init_buffer() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_buf_initialized) return;
    g_write_buf.resize(WRITE_BUF_SIZE);
    g_write_len = 0;
    g_buf_initialized = true;
}

static void flush_buffer() {
    if (g_fp && g_write_len > 0) {
        fwrite(g_write_buf.data(), 1, g_write_len, g_fp);
        g_write_len = 0;
    }
}

static void append_line(const char* line) {
    size_t n = strlen(line);
    if (n + 1 > WRITE_BUF_SIZE) return;  // skip oversized
    init_buffer();
    if (g_write_len + n + 2 > WRITE_BUF_SIZE) {
        flush_buffer();
    }
    memcpy(g_write_buf.data() + g_write_len, line, n);
    g_write_len += n;
    g_write_buf[g_write_len++] = '\n';
}

static void open_output_file_if_needed() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_fp) return;

    const char* trace_dir = getenv("TELLER_TRACE_DIR");
    if (!trace_dir || !trace_dir[0]) {
        const char* data_dir = getenv("TELLER_DATA_DIR");
        if (data_dir && data_dir[0]) {
            static char buf[512];
            snprintf(buf, sizeof(buf), "%s/trace", data_dir);
            trace_dir = buf;
        } else {
            trace_dir = "/data/trace";
        }
    }

    if (mkdir(trace_dir, 0777) == -1 && errno != EEXIST) {
        fprintf(stderr, "[teller] error: Failed to mkdir %s: %s\n", trace_dir, strerror(errno));
        return;
    }

    char filename[512];
    snprintf(filename, sizeof(filename), "%s/output_pid%d.tmp.jsonl", trace_dir, getpid());
    g_fp = fopen(filename, "a");
    if (!g_fp) {
        fprintf(stderr, "[teller] error: Failed to open %s: %s\n", filename, strerror(errno));
    }
}

static void CUPTIAPI activityBufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *size = 16 * 1024;
    *buffer = (uint8_t *)malloc(*size);
    *maxNumRecords = 0;
}

static void write_json(const char* json_str) {
    if (g_fp) append_line(json_str);
}

// CUDA 11.2+: CUpti_ActivityKernel9. For older CUDA, change to Kernel8 and rebuild.
void CUPTIAPI activityBufferCompleted(CUcontext ctx, uint32_t streamId,
                                      uint8_t* buffer, size_t size, size_t validSize) {
    CUpti_Activity* record = NULL;
    CUptiResult status;

    open_output_file_if_needed();

    while ((status = cuptiActivityGetNextRecord(buffer, validSize, &record)) == CUPTI_SUCCESS) {
        char json_buf[4096];
        switch (record->kind) {
            case CUPTI_ACTIVITY_KIND_MARKER: {
                CUpti_ActivityMarker2* marker = (CUpti_ActivityMarker2*)record;
                uint32_t processId = marker->objectId.pt.processId;
                uint32_t threadId = marker->objectId.pt.threadId;
                /* Range start (push) has name; range end (pop) has name==NULL from CUPTI. */
                const char* name = marker->name ? marker->name : "[range_end]";
                snprintf(json_buf, sizeof(json_buf),
                    "{\"type\":\"NVTX_MARKER\",\"name\":\"%s\",\"timestamp\":%lu,\"id\":%u,\"process_id\":%u,\"thread_id\":%u}",
                    name, (unsigned long)marker->timestamp, marker->id, processId, threadId);
                write_json(json_buf);
                break;
            }
            case CUPTI_ACTIVITY_KIND_KERNEL: {
                CUpti_ActivityKernel9* kernel = (CUpti_ActivityKernel9*)record;
                unsigned long start = (unsigned long)kernel->start;
                unsigned long end = (unsigned long)kernel->end;
                unsigned long dur = end - start;
                const char* n = kernel->name ? kernel->name : "";
                snprintf(json_buf, sizeof(json_buf),
                    "{\"t\":\"KERNEL\",\"n\":\"%s\",\"gs\":%lu,\"ge\":%lu,\"d\":%lu,\"c\":%u}",
                    n, start, end, dur, kernel->correlationId);
                write_json(json_buf);
                break;
            }
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
                CUpti_ActivityKernel9* kernel = (CUpti_ActivityKernel9*)record;
                unsigned long start = (unsigned long)kernel->start;
                unsigned long end = (unsigned long)kernel->end;
                unsigned long dur = end - start;
                const char* n = kernel->name ? kernel->name : "";
                snprintf(json_buf, sizeof(json_buf),
                    "{\"t\":\"CONCURRENT_KERNEL\",\"n\":\"%s\",\"gs\":%lu,\"ge\":%lu,\"d\":%lu,\"c\":%u}",
                    n, start, end, dur, kernel->correlationId);
                write_json(json_buf);
                break;
            }
            case CUPTI_ACTIVITY_KIND_RUNTIME: {
                CUpti_ActivityAPI* runtime = (CUpti_ActivityAPI*)record;
                const char* funcName = nullptr;
                cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, runtime->cbid, &funcName);
                if (!funcName) funcName = "Unknown";
                if (has_allowed_prefix(funcName)) {
                    unsigned long s = (unsigned long)runtime->start;
                    unsigned long e = (unsigned long)runtime->end;
                    unsigned long d = e - s;
                    snprintf(json_buf, sizeof(json_buf),
                        "{\"t\":\"RUNTIME\",\"cbid\":%u,\"n\":\"%s\",\"s\":%lu,\"e\":%lu,\"d\":%lu,\"c\":%u,\"pid\":%u,\"tid\":%u}",
                        runtime->cbid, funcName, s, e, d, runtime->correlationId, runtime->processId, runtime->threadId);
                    write_json(json_buf);
                }
                break;
            }
            case CUPTI_ACTIVITY_KIND_DRIVER: {
                CUpti_ActivityAPI* driver = (CUpti_ActivityAPI*)record;
                const char* funcName = nullptr;
                cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, driver->cbid, &funcName);
                if (!funcName) funcName = "Unknown";
                if (strncmp(funcName, "cuLaunchKernel", 14) == 0) {
                    unsigned long s = (unsigned long)driver->start;
                    unsigned long e = (unsigned long)driver->end;
                    unsigned long d = e - s;
                    snprintf(json_buf, sizeof(json_buf),
                        "{\"t\":\"DRIVER\",\"cbid\":%u,\"n\":\"%s\",\"s\":%lu,\"e\":%lu,\"d\":%lu,\"c\":%u,\"pid\":%u,\"tid\":%u}",
                        driver->cbid, funcName, s, e, d, driver->correlationId, driver->processId, driver->threadId);
                    write_json(json_buf);
                }
                break;
            }
            default:
                break;
        }
    }

    if (status != CUPTI_SUCCESS && status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
        const char* errstr;
        cuptiGetResultString(status, &errstr);
        fprintf(stderr, "[teller] error: CUPTI %s\n", errstr);
    }
    free(buffer);
}

extern "C" int InitializeInjection() {
    if (tracingEnabled) return 1;
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER);
    cuptiActivityRegisterCallbacks(activityBufferRequested, activityBufferCompleted);
    tracingEnabled = true;
    return 1;
}

__attribute__((constructor))
static void init() {
    InitializeInjection();
}

__attribute__((destructor))
static void fini() {
    if (tracingEnabled) {
        cuptiActivityFlushAll(0);
        flush_buffer();
        if (g_fp) { fclose(g_fp); g_fp = nullptr; }
    }
}
