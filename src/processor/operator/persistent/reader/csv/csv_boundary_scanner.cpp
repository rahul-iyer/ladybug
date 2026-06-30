#include "processor/operator/persistent/reader/csv/csv_boundary_scanner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <thread>

#include "common/constants.h"
#include "common/file_system/virtual_file_system.h"
#include "main/client_context.h"
#include <bit>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace lbug {
namespace processor {

using namespace common;

namespace {

enum class CSVBoundaryScannerCharClass : uint8_t {
    Other,
    Delimiter,
    Quote,
    Escape,
    CarriageReturn,
    LineFeed,
    COUNT,
};

enum CSVBoundaryScannerAction : uint8_t {
    None = 0,
    MarkQuotedMultiline = 1u << 0u,
    MarkLogicalBoundary = 1u << 1u,
    SetFieldStartTrue = 1u << 2u,
    SetFieldStartFalse = 1u << 3u,
    MarkInvalid = 1u << 4u,
};

struct CSVBoundaryScannerTransition {
    CSVBoundaryScannerState nextState;
    uint8_t actions;
};

struct CSVBoundaryScannerMasks {
    uint32_t interesting = 0;
};

enum class CSVBoundaryScannerSeed : uint8_t {
    OutsideFieldStart,
    OutsideFieldMiddle,
    InQuotedField,
    AfterQuote,
    Escaped,
    CarriageReturn,
    COUNT,
};

struct CSVBoundaryScannerTransitionResult {
    CSVBoundaryScannerSeed outSeed = CSVBoundaryScannerSeed::OutsideFieldStart;
    CSVBoundaryScannerSeed overlapOutSeed = CSVBoundaryScannerSeed::OutsideFieldStart;
    bool hasBoundary = false;
    uint64_t firstBoundaryOffset = 0;
    uint64_t lastBoundaryOffset = 0;
    uint64_t maxClosedRowLength = 0;
    bool detectedQuotedMultiline = false;
    bool sawInvalidQuotedTransition = false;
};

struct CSVBoundaryChunkSummary {
    std::array<CSVBoundaryScannerTransitionResult,
        static_cast<uint8_t>(CSVBoundaryScannerSeed::COUNT)>
        transitions;
};

struct RangeAccumulator {
    explicit RangeAccumulator(idx_t fileIdx) : fileIdx{fileIdx} {}

    void finalizeBoundary(uint64_t boundaryOffset) {
        detectedOversizedLogicalRow =
            detectedOversizedLogicalRow ||
            boundaryOffset - currentLogicalRowStart > CopyConstants::PARALLEL_BLOCK_SIZE;
        currentLogicalRowStart = boundaryOffset;
        if (boundaryOffset - currentRangeStart < CopyConstants::PARALLEL_BLOCK_SIZE) {
            return;
        }
        ranges.push_back(CSVParseRange{fileIdx, currentRangeStart, boundaryOffset, nextRangeIdx++,
            currentRangeStart == 0});
        currentRangeStart = boundaryOffset;
    }

    void finalizeEOF(uint64_t fileSize) {
        detectedOversizedLogicalRow =
            detectedOversizedLogicalRow ||
            fileSize - currentLogicalRowStart > CopyConstants::PARALLEL_BLOCK_SIZE;
        if (fileSize > currentRangeStart) {
            ranges.push_back(CSVParseRange{fileIdx, currentRangeStart, fileSize, nextRangeIdx++,
                currentRangeStart == 0});
        }
    }

    idx_t fileIdx;
    uint64_t currentRangeStart = 0;
    uint64_t currentLogicalRowStart = 0;
    block_idx_t nextRangeIdx = 0;
    bool detectedOversizedLogicalRow = false;
    std::vector<CSVParseRange> ranges;
};

struct CSVBoundaryScannerRuntimeState {
    CSVBoundaryScannerState state = CSVBoundaryScannerState::OutsideField;
    bool atFieldStart = true;
    uint64_t pendingCarriageReturnOffset = 0;
    bool sawInvalidQuotedTransition = false;
};

CSVBoundaryScannerRuntimeState runtimeStateFromSeed(CSVBoundaryScannerSeed seed,
    uint64_t chunkOffset) {
    CSVBoundaryScannerRuntimeState state;
    switch (seed) {
    case CSVBoundaryScannerSeed::OutsideFieldStart:
        state.state = CSVBoundaryScannerState::OutsideField;
        state.atFieldStart = true;
        break;
    case CSVBoundaryScannerSeed::OutsideFieldMiddle:
        state.state = CSVBoundaryScannerState::OutsideField;
        state.atFieldStart = false;
        break;
    case CSVBoundaryScannerSeed::InQuotedField:
        state.state = CSVBoundaryScannerState::InQuotedField;
        state.atFieldStart = false;
        break;
    case CSVBoundaryScannerSeed::AfterQuote:
        state.state = CSVBoundaryScannerState::AfterQuote;
        state.atFieldStart = false;
        break;
    case CSVBoundaryScannerSeed::Escaped:
        state.state = CSVBoundaryScannerState::Escaped;
        state.atFieldStart = false;
        break;
    case CSVBoundaryScannerSeed::CarriageReturn:
        state.state = CSVBoundaryScannerState::CarriageReturn;
        state.atFieldStart = true;
        state.pendingCarriageReturnOffset = chunkOffset == 0 ? 0 : chunkOffset - 1;
        break;
    case CSVBoundaryScannerSeed::COUNT:
        UNREACHABLE_CODE;
    }
    return state;
}

CSVBoundaryScannerSeed seedFromRuntimeState(const CSVBoundaryScannerRuntimeState& state) {
    switch (state.state) {
    case CSVBoundaryScannerState::OutsideField:
        return state.atFieldStart ? CSVBoundaryScannerSeed::OutsideFieldStart :
                                    CSVBoundaryScannerSeed::OutsideFieldMiddle;
    case CSVBoundaryScannerState::InQuotedField:
        return CSVBoundaryScannerSeed::InQuotedField;
    case CSVBoundaryScannerState::AfterQuote:
        return CSVBoundaryScannerSeed::AfterQuote;
    case CSVBoundaryScannerState::Escaped:
        return CSVBoundaryScannerSeed::Escaped;
    case CSVBoundaryScannerState::CarriageReturn:
        return CSVBoundaryScannerSeed::CarriageReturn;
    }
    UNREACHABLE_CODE;
}

class CSVBoundaryScannerFSM {
public:
    explicit CSVBoundaryScannerFSM(const CSVOption& option) : classTable{buildClassTable(option)} {}

    template<typename BoundaryFunc>
    void consumeBoringSpan(uint64_t spanLength, BoundaryFunc&& onBoundary,
        CSVBoundaryScannerRuntimeState& runtimeState) const {
        if (spanLength == 0) {
            return;
        }
        switch (runtimeState.state) {
        case CSVBoundaryScannerState::OutsideField:
            runtimeState.atFieldStart = false;
            break;
        case CSVBoundaryScannerState::InQuotedField:
            break;
        case CSVBoundaryScannerState::AfterQuote:
            runtimeState.sawInvalidQuotedTransition = true;
            runtimeState.state = CSVBoundaryScannerState::OutsideField;
            runtimeState.atFieldStart = false;
            break;
        case CSVBoundaryScannerState::Escaped:
            runtimeState.sawInvalidQuotedTransition = true;
            runtimeState.state = CSVBoundaryScannerState::InQuotedField;
            break;
        case CSVBoundaryScannerState::CarriageReturn:
            onBoundary(runtimeState.pendingCarriageReturnOffset + 1);
            runtimeState.state = CSVBoundaryScannerState::OutsideField;
            runtimeState.atFieldStart = false;
            break;
        }
    }

    template<typename BoundaryFunc>
    void step(char c, uint64_t absoluteOffset, BoundaryFunc&& onBoundary,
        bool& detectedQuotedMultiline, CSVBoundaryScannerRuntimeState& runtimeState) const {
        auto cls = getCharClass(c);
        if (runtimeState.state == CSVBoundaryScannerState::CarriageReturn &&
            cls == CSVBoundaryScannerCharClass::LineFeed) {
            onBoundary(absoluteOffset + 1);
            runtimeState.state = CSVBoundaryScannerState::OutsideField;
            runtimeState.atFieldStart = true;
            return;
        }
        if (runtimeState.state == CSVBoundaryScannerState::OutsideField &&
            cls == CSVBoundaryScannerCharClass::Quote) {
            if (runtimeState.atFieldStart) {
                runtimeState.state = CSVBoundaryScannerState::InQuotedField;
                runtimeState.atFieldStart = false;
            } else {
                runtimeState.atFieldStart = false;
            }
            return;
        }
        if (runtimeState.state == CSVBoundaryScannerState::CarriageReturn) {
            onBoundary(runtimeState.pendingCarriageReturnOffset + 1);
            runtimeState.state = CSVBoundaryScannerState::OutsideField;
        }

        const auto transition =
            TRANSITIONS[static_cast<uint8_t>(runtimeState.state)][static_cast<uint8_t>(cls)];
        runtimeState.state = transition.nextState;
        applyActions(transition.actions, absoluteOffset, onBoundary, detectedQuotedMultiline,
            runtimeState);
    }

private:
    static constexpr std::array<std::array<CSVBoundaryScannerTransition,
                                    static_cast<uint8_t>(CSVBoundaryScannerCharClass::COUNT)>,
        5>
        TRANSITIONS{{
            // OutsideField
            {{{CSVBoundaryScannerState::OutsideField, SetFieldStartFalse},
                {CSVBoundaryScannerState::OutsideField, SetFieldStartTrue},
                {CSVBoundaryScannerState::OutsideField, None},
                {CSVBoundaryScannerState::OutsideField, SetFieldStartFalse},
                {CSVBoundaryScannerState::CarriageReturn, SetFieldStartTrue},
                {CSVBoundaryScannerState::OutsideField, MarkLogicalBoundary | SetFieldStartTrue}}},
            // InQuotedField
            {{{CSVBoundaryScannerState::InQuotedField, None},
                {CSVBoundaryScannerState::InQuotedField, None},
                {CSVBoundaryScannerState::AfterQuote, None},
                {CSVBoundaryScannerState::Escaped, None},
                {CSVBoundaryScannerState::InQuotedField, MarkQuotedMultiline},
                {CSVBoundaryScannerState::InQuotedField, MarkQuotedMultiline}}},
            // AfterQuote
            {{{CSVBoundaryScannerState::OutsideField, MarkInvalid | SetFieldStartFalse},
                {CSVBoundaryScannerState::OutsideField, SetFieldStartTrue},
                {CSVBoundaryScannerState::InQuotedField, None},
                {CSVBoundaryScannerState::OutsideField, MarkInvalid | SetFieldStartFalse},
                {CSVBoundaryScannerState::CarriageReturn, SetFieldStartTrue},
                {CSVBoundaryScannerState::OutsideField, MarkLogicalBoundary | SetFieldStartTrue}}},
            // Escaped
            {{{CSVBoundaryScannerState::InQuotedField, MarkInvalid},
                {CSVBoundaryScannerState::InQuotedField, MarkInvalid},
                {CSVBoundaryScannerState::InQuotedField, None},
                {CSVBoundaryScannerState::InQuotedField, None},
                {CSVBoundaryScannerState::InQuotedField, MarkInvalid | MarkQuotedMultiline},
                {CSVBoundaryScannerState::InQuotedField, MarkInvalid | MarkQuotedMultiline}}},
            // CarriageReturn
            {{{CSVBoundaryScannerState::OutsideField, SetFieldStartFalse},
                {CSVBoundaryScannerState::OutsideField, SetFieldStartTrue},
                {CSVBoundaryScannerState::OutsideField, None},
                {CSVBoundaryScannerState::OutsideField, SetFieldStartFalse},
                {CSVBoundaryScannerState::CarriageReturn, SetFieldStartTrue},
                {CSVBoundaryScannerState::OutsideField, MarkLogicalBoundary | SetFieldStartTrue}}},
        }};

    static std::array<CSVBoundaryScannerCharClass, 256> buildClassTable(const CSVOption& option) {
        std::array<CSVBoundaryScannerCharClass, 256> table;
        table.fill(CSVBoundaryScannerCharClass::Other);
        table[static_cast<uint8_t>('\r')] = CSVBoundaryScannerCharClass::CarriageReturn;
        table[static_cast<uint8_t>('\n')] = CSVBoundaryScannerCharClass::LineFeed;
        table[static_cast<uint8_t>(option.delimiter)] = CSVBoundaryScannerCharClass::Delimiter;
        table[static_cast<uint8_t>(option.escapeChar)] = CSVBoundaryScannerCharClass::Escape;
        table[static_cast<uint8_t>(option.quoteChar)] = CSVBoundaryScannerCharClass::Quote;
        return table;
    }

    CSVBoundaryScannerCharClass getCharClass(char c) const {
        return classTable[static_cast<uint8_t>(c)];
    }

    template<typename BoundaryFunc>
    static void applyActions(uint8_t actions, uint64_t absoluteOffset, BoundaryFunc&& onBoundary,
        bool& detectedQuotedMultiline, CSVBoundaryScannerRuntimeState& runtimeState) {
        if ((actions & MarkQuotedMultiline) != 0) {
            detectedQuotedMultiline = true;
        }
        if ((actions & MarkLogicalBoundary) != 0) {
            onBoundary(absoluteOffset + 1);
        }
        if ((actions & SetFieldStartTrue) != 0) {
            runtimeState.atFieldStart = true;
        } else if ((actions & SetFieldStartFalse) != 0) {
            runtimeState.atFieldStart = false;
        }
        if ((actions & MarkInvalid) != 0) {
            runtimeState.sawInvalidQuotedTransition = true;
        }
        if (runtimeState.state == CSVBoundaryScannerState::CarriageReturn) {
            runtimeState.pendingCarriageReturnOffset = absoluteOffset;
        }
    }

    std::array<CSVBoundaryScannerCharClass, 256> classTable;
};

CSVBoundaryScannerMasks scanScalar(const char* buffer, uint64_t bytesToRead,
    const CSVOption& option) {
    CSVBoundaryScannerMasks masks;
    for (uint64_t i = 0; i < bytesToRead; ++i) {
        const auto c = buffer[i];
        if (c == option.delimiter || c == option.quoteChar || c == option.escapeChar || c == '\r' ||
            c == '\n') {
            masks.interesting |= (1u << i);
        }
    }
    return masks;
}

#if defined(__SSE2__)
CSVBoundaryScannerMasks scanSSE2(const char* buffer, const CSVOption& option) {
    const auto data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buffer));
    auto mask = _mm_or_si128(_mm_cmpeq_epi8(data, _mm_set1_epi8(option.delimiter)),
        _mm_cmpeq_epi8(data, _mm_set1_epi8(option.quoteChar)));
    mask = _mm_or_si128(mask, _mm_cmpeq_epi8(data, _mm_set1_epi8(option.escapeChar)));
    mask = _mm_or_si128(mask, _mm_cmpeq_epi8(data, _mm_set1_epi8('\r')));
    mask = _mm_or_si128(mask, _mm_cmpeq_epi8(data, _mm_set1_epi8('\n')));
    return {static_cast<uint32_t>(_mm_movemask_epi8(mask))};
}
#endif

#if defined(__ARM_NEON)
CSVBoundaryScannerMasks scanNEON(const char* buffer, const CSVOption& option) {
    const auto data = vld1q_u8(reinterpret_cast<const uint8_t*>(buffer));
    auto mask = vorrq_u8(vceqq_u8(data, vdupq_n_u8(static_cast<uint8_t>(option.delimiter))),
        vceqq_u8(data, vdupq_n_u8(static_cast<uint8_t>(option.quoteChar))));
    mask = vorrq_u8(mask, vceqq_u8(data, vdupq_n_u8(static_cast<uint8_t>(option.escapeChar))));
    mask = vorrq_u8(mask, vceqq_u8(data, vdupq_n_u8(static_cast<uint8_t>('\r'))));
    mask = vorrq_u8(mask, vceqq_u8(data, vdupq_n_u8(static_cast<uint8_t>('\n'))));
    alignas(16) uint8_t bytes[16];
    vst1q_u8(bytes, mask);
    uint32_t interesting = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        if (bytes[i] != 0) {
            interesting |= (1u << i);
        }
    }
    return {interesting};
}
#endif

CSVBoundaryScannerMasks scanInterestingBytes(const char* buffer, uint64_t bytesToRead,
    const CSVOption& option) {
#if defined(__SSE2__)
    if (bytesToRead == 16) {
        return scanSSE2(buffer, option);
    }
#elif defined(__ARM_NEON)
    if (bytesToRead == 16) {
        return scanNEON(buffer, option);
    }
#endif
    return scanScalar(buffer, bytesToRead, option);
}

template<typename BoundaryFunc>
void processChunk(const char* buffer, uint64_t bytesToRead, uint64_t chunkOffset,
    const CSVOption& option, const CSVBoundaryScannerFSM& fsm,
    CSVBoundaryScannerRuntimeState& runtimeState, bool& detectedQuotedMultiline,
    BoundaryFunc&& onBoundary) {
    static constexpr uint64_t SIMD_WIDTH = 16;
    uint64_t i = 0;
    while (i < bytesToRead) {
        const auto laneWidth = std::min<uint64_t>(SIMD_WIDTH, bytesToRead - i);
        const auto masks = scanInterestingBytes(buffer + i, laneWidth, option);
        if (masks.interesting == 0) {
            fsm.consumeBoringSpan(laneWidth, onBoundary, runtimeState);
            i += laneWidth;
            continue;
        }

        uint64_t lastProcessed = 0;
        auto interesting = masks.interesting;
        while (interesting != 0) {
            const auto bit = std::countr_zero(interesting);
            fsm.consumeBoringSpan(bit - lastProcessed, onBoundary, runtimeState);
            const auto absoluteOffset = chunkOffset + i + bit;
            fsm.step(buffer[i + bit], absoluteOffset, onBoundary, detectedQuotedMultiline,
                runtimeState);
            lastProcessed = bit + 1;
            interesting &= interesting - 1;
        }
        fsm.consumeBoringSpan(laneWidth - lastProcessed, onBoundary, runtimeState);
        i += laneWidth;
    }
}

CSVBoundaryScannerTransitionResult scanTransition(const char* buffer, uint64_t bytesToRead,
    uint64_t nominalBytesToRead, uint64_t chunkOffset, const CSVOption& option,
    const CSVBoundaryScannerFSM& fsm, CSVBoundaryScannerSeed inSeed) {
    CSVBoundaryScannerTransitionResult result;
    auto runtimeState = runtimeStateFromSeed(inSeed, chunkOffset);
    uint64_t previousBoundary = chunkOffset;
    auto onBoundary = [&result, &previousBoundary](uint64_t boundaryOffset) {
        if (!result.hasBoundary) {
            result.firstBoundaryOffset = boundaryOffset;
            result.hasBoundary = true;
        }
        result.maxClosedRowLength =
            std::max<uint64_t>(result.maxClosedRowLength, boundaryOffset - previousBoundary);
        result.lastBoundaryOffset = boundaryOffset;
        previousBoundary = boundaryOffset;
    };
    processChunk(buffer, nominalBytesToRead, chunkOffset, option, fsm, runtimeState,
        result.detectedQuotedMultiline, onBoundary);
    result.outSeed = seedFromRuntimeState(runtimeState);
    if (bytesToRead > nominalBytesToRead) {
        processChunk(buffer + nominalBytesToRead, bytesToRead - nominalBytesToRead,
            chunkOffset + nominalBytesToRead, option, fsm, runtimeState,
            result.detectedQuotedMultiline, onBoundary);
    }
    result.overlapOutSeed = seedFromRuntimeState(runtimeState);
    result.sawInvalidQuotedTransition = runtimeState.sawInvalidQuotedTransition;
    return result;
}

CSVBoundaryScannerTransitionResult scanUntilBoundary(FileInfo* fileInfo, uint64_t fileSize,
    uint64_t offset, uint64_t previousBoundary, const CSVOption& option,
    CSVBoundaryScannerSeed inSeed) {
    static constexpr uint64_t MAX_EXTENSION_SIZE = 1u << 20;
    CSVBoundaryScannerTransitionResult result;
    if (offset >= fileSize) {
        result.outSeed = inSeed;
        result.overlapOutSeed = inSeed;
        return result;
    }
    CSVBoundaryScannerFSM fsm{option};
    auto runtimeState = runtimeStateFromSeed(inSeed, offset);
    auto extensionSize = CopyConstants::PARALLEL_BLOCK_SIZE;
    auto onBoundary = [&result, &previousBoundary](uint64_t boundaryOffset) {
        if (!result.hasBoundary) {
            result.firstBoundaryOffset = boundaryOffset;
            result.hasBoundary = true;
        }
        result.maxClosedRowLength =
            std::max<uint64_t>(result.maxClosedRowLength, boundaryOffset - previousBoundary);
        result.lastBoundaryOffset = boundaryOffset;
        previousBoundary = boundaryOffset;
    };
    while (offset < fileSize && !result.hasBoundary) {
        const auto bytesToRead = std::min<uint64_t>(extensionSize, fileSize - offset);
        auto buffer = std::make_unique<char[]>(bytesToRead);
        fileInfo->readFromFile(buffer.get(), bytesToRead, offset);
        processChunk(buffer.get(), bytesToRead, offset, option, fsm, runtimeState,
            result.detectedQuotedMultiline, onBoundary);
        offset += bytesToRead;
        extensionSize = std::min<uint64_t>(extensionSize * 2, MAX_EXTENSION_SIZE);
    }
    result.outSeed = seedFromRuntimeState(runtimeState);
    result.overlapOutSeed = result.outSeed;
    result.sawInvalidQuotedTransition = runtimeState.sawInvalidQuotedTransition;
    return result;
}

template<typename Func>
void parallelFor(uint64_t count, uint64_t numThreads, Func&& func) {
    if (count == 0) {
        return;
    }
    numThreads = std::max<uint64_t>(1, std::min<uint64_t>(numThreads, count));
    std::atomic<uint64_t> nextIdx = 0;
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (uint64_t i = 0; i < numThreads; ++i) {
        threads.emplace_back([&]() {
            while (true) {
                const auto idx = nextIdx.fetch_add(1);
                if (idx >= count) {
                    return;
                }
                func(idx);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

} // namespace

CSVBoundaryScanResult CSVBoundaryScanner::scanFile(const std::string& filePath, idx_t fileIdx,
    const CSVOption& option, main::ClientContext* context) {
    auto fileInfo = VirtualFileSystem::GetUnsafe(*context)->openFile(filePath,
        FileOpenFlags(FileFlags::READ_ONLY
#ifdef _WIN32
                      | FileFlags::BINARY
#endif
            ),
        context);
    CSVBoundaryScanResult result;
    result.fileSize = fileInfo->getFileSize();
    if (result.fileSize == 0) {
        return result;
    }

    static constexpr uint64_t SCAN_CHUNK_SIZE = 1u << 20;
    static constexpr uint64_t INITIAL_OVERLAP_SIZE = 1u << 10;
    static constexpr uint64_t PLANNED_RANGE_TARGET_SIZE = SCAN_CHUNK_SIZE;
    const auto numChunks = (result.fileSize + SCAN_CHUNK_SIZE - 1) / SCAN_CHUNK_SIZE;
    const auto numThreads = std::max<uint64_t>(1, context->getMaxNumThreadForExec());
    std::vector<CSVBoundaryChunkSummary> chunkSummaries(numChunks);

    parallelFor(numChunks, numThreads, [&](uint64_t chunkIdx) {
        auto localFileInfo = VirtualFileSystem::GetUnsafe(*context)->openFile(filePath,
            FileOpenFlags(FileFlags::READ_ONLY
#ifdef _WIN32
                          | FileFlags::BINARY
#endif
                ),
            context);
        const auto chunkOffset = chunkIdx * SCAN_CHUNK_SIZE;
        const auto nominalBytesToRead =
            std::min<uint64_t>(SCAN_CHUNK_SIZE, result.fileSize - chunkOffset);
        const auto bytesToRead = std::min<uint64_t>(nominalBytesToRead + INITIAL_OVERLAP_SIZE,
            result.fileSize - chunkOffset);
        auto buffer = std::make_unique<char[]>(bytesToRead);
        localFileInfo->readFromFile(buffer.get(), bytesToRead, chunkOffset);
        CSVBoundaryScannerFSM fsm{option};
        for (auto seedIdx = 0u; seedIdx < static_cast<uint8_t>(CSVBoundaryScannerSeed::COUNT);
             ++seedIdx) {
            chunkSummaries[chunkIdx].transitions[seedIdx] =
                scanTransition(buffer.get(), bytesToRead, nominalBytesToRead, chunkOffset, option,
                    fsm, static_cast<CSVBoundaryScannerSeed>(seedIdx));
        }
    });

    RangeAccumulator rangeAccumulator{fileIdx};
    uint64_t currentLogicalRowStart = 0;
    bool sawInvalidQuotedTransition = false;
    auto currentSeed = CSVBoundaryScannerSeed::OutsideFieldStart;
    for (uint64_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
        auto replayResult = chunkSummaries[chunkIdx].transitions[static_cast<uint8_t>(currentSeed)];
        currentSeed = replayResult.outSeed;
        const auto chunkOffset = chunkIdx * SCAN_CHUNK_SIZE;
        const auto nominalEndOffset =
            std::min<uint64_t>(result.fileSize, chunkOffset + SCAN_CHUNK_SIZE);
        const auto overlappedEndOffset =
            std::min<uint64_t>(result.fileSize, nominalEndOffset + INITIAL_OVERLAP_SIZE);
        const auto rangeTargetOffset =
            rangeAccumulator.currentRangeStart + PLANNED_RANGE_TARGET_SIZE;
        const auto shouldExtend =
            rangeTargetOffset <= nominalEndOffset &&
            (!replayResult.hasBoundary || replayResult.lastBoundaryOffset < rangeTargetOffset) &&
            overlappedEndOffset < result.fileSize;
        if (shouldExtend) {
            const auto extensionPreviousBoundary =
                replayResult.hasBoundary ? replayResult.lastBoundaryOffset : currentLogicalRowStart;
            auto extensionResult =
                scanUntilBoundary(fileInfo.get(), result.fileSize, overlappedEndOffset,
                    extensionPreviousBoundary, option, replayResult.overlapOutSeed);
            sawInvalidQuotedTransition =
                sawInvalidQuotedTransition || extensionResult.sawInvalidQuotedTransition;
            result.detectedQuotedMultiline =
                result.detectedQuotedMultiline || extensionResult.detectedQuotedMultiline;
            if (extensionResult.hasBoundary) {
                if (!replayResult.hasBoundary) {
                    replayResult.firstBoundaryOffset = extensionResult.firstBoundaryOffset;
                }
                replayResult.hasBoundary = true;
                replayResult.lastBoundaryOffset = extensionResult.lastBoundaryOffset;
                replayResult.maxClosedRowLength =
                    std::max(replayResult.maxClosedRowLength, extensionResult.maxClosedRowLength);
            }
        }
        sawInvalidQuotedTransition =
            sawInvalidQuotedTransition || replayResult.sawInvalidQuotedTransition;
        result.detectedQuotedMultiline =
            result.detectedQuotedMultiline || replayResult.detectedQuotedMultiline;
        if (!replayResult.hasBoundary ||
            replayResult.lastBoundaryOffset <= currentLogicalRowStart) {
            continue;
        }
        const auto firstNewBoundaryOffset =
            std::max(replayResult.firstBoundaryOffset, currentLogicalRowStart);
        rangeAccumulator.detectedOversizedLogicalRow =
            rangeAccumulator.detectedOversizedLogicalRow ||
            firstNewBoundaryOffset - currentLogicalRowStart > CopyConstants::PARALLEL_BLOCK_SIZE ||
            replayResult.maxClosedRowLength > CopyConstants::PARALLEL_BLOCK_SIZE;
        currentLogicalRowStart = replayResult.lastBoundaryOffset;
        if (replayResult.lastBoundaryOffset > rangeAccumulator.currentRangeStart &&
            replayResult.lastBoundaryOffset - rangeAccumulator.currentRangeStart >=
                PLANNED_RANGE_TARGET_SIZE) {
            rangeAccumulator.ranges.push_back(CSVParseRange{fileIdx,
                rangeAccumulator.currentRangeStart, replayResult.lastBoundaryOffset,
                rangeAccumulator.nextRangeIdx++, rangeAccumulator.currentRangeStart == 0});
            rangeAccumulator.currentRangeStart = replayResult.lastBoundaryOffset;
        }
    }

    sawInvalidQuotedTransition = sawInvalidQuotedTransition ||
                                 currentSeed == CSVBoundaryScannerSeed::InQuotedField ||
                                 currentSeed == CSVBoundaryScannerSeed::Escaped;
    rangeAccumulator.detectedOversizedLogicalRow =
        rangeAccumulator.detectedOversizedLogicalRow ||
        result.fileSize - currentLogicalRowStart > CopyConstants::PARALLEL_BLOCK_SIZE;
    if (result.fileSize > rangeAccumulator.currentRangeStart) {
        rangeAccumulator.ranges.push_back(
            CSVParseRange{fileIdx, rangeAccumulator.currentRangeStart, result.fileSize,
                rangeAccumulator.nextRangeIdx++, rangeAccumulator.currentRangeStart == 0});
    }
    result.ranges = std::move(rangeAccumulator.ranges);
    result.detectedOversizedLogicalRow = rangeAccumulator.detectedOversizedLogicalRow;
    result.usePlannedRanges =
        (result.detectedQuotedMultiline || result.detectedOversizedLogicalRow) &&
        !sawInvalidQuotedTransition && !result.ranges.empty();
    if (!result.usePlannedRanges) {
        result.ranges.clear();
    }
    return result;
}

} // namespace processor
} // namespace lbug
