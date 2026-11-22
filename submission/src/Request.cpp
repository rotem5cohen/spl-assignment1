#include "Request.h"
#include "Model.h"
#include "GPU.h"
#include <memory>

Request::Request(int id,
                 const Model& model,
                 int admissionCycle,
                 std::string prompt)
    : id(id),
      model(model),
      admissionCycle(admissionCycle),
      tokens(model.tokenizer(prompt)),
      memoryBlocks(),
      reservedBlocks(0) {
}

int Request::getId() const {
    return id;
}

const Model& Request::getModel() const {
    return model;
}

int Request::getAdmissionCycle() const {
    return admissionCycle;
}
Request::~Request() {
}

void Request::deallocateMemory(GPU& gpu) {
    gpu.unreserve(getNeedToUnreserve(), *this);

    for (std::size_t slot : memoryBlocks) {
        gpu.release(slot);
    }

    memoryBlocks.clear();
    reservedBlocks = 0;
}

GenerativeRequest::GenerativeRequest(
    int id,
    const Model& model,
    int admissionCycle,
    std::string prompt,
    std::string scriptedOutput,
    std::size_t maxOutputTokens)

    : Request(id, model, admissionCycle, prompt),
      scriptedOutput(scriptedOutput),
      maxOutputTokens(maxOutputTokens),
      outputTokens(model.tokenizer(scriptedOutput)),
      processedTokens(0),
      generatedTokens(0) {
}

bool GenerativeRequest::finished() const {
    return processedTokens == tokens.size() &&
           (generatedTokens == maxOutputTokens ||
            generatedTokens == outputTokens.size());
}

std::size_t GenerativeRequest::getMaxNeededMemory() const {
    return (tokens.size() + maxOutputTokens) *
           model.getEmbeddingSize();
}

std::size_t GenerativeRequest::getNeededMemory(
    const GPU& gpu) const {

    PrefixMatch match = gpu.findLongestCachedPrefix(
        model.getModelId(),
        model.getEmbeddingSize(),
        tokens
    );

    return (tokens.size() - match.matchedTokens +
            maxOutputTokens) * model.getEmbeddingSize();
}

std::vector<std::string>
GenerativeRequest::getCacheableTokens() const {
    return tokens;
}

void GenerativeRequest::reserveMemory(GPU& gpu) {

    PrefixMatch match = gpu.findLongestCachedPrefix(
        model.getModelId(),
        model.getEmbeddingSize(),
        tokens
    );

    for (std::size_t slot : match.blockIds) {
        gpu.acquire(slot);
        memoryBlocks.push_back(slot);
    }

    processedTokens = match.matchedTokens;

    reservedBlocks = gpu.getBlocksForMemory(
        getNeededMemory(gpu)
    );

    gpu.reserve(reservedBlocks, *this);
}

std::size_t GenerativeRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}

void GenerativeRequest::act(GPU& gpu) {
    if (finished()) {
        return;
    }

    std::size_t embeddingSize = model.getEmbeddingSize();
    std::size_t tokensPerBlock =
        gpu.getBlockSize() / embeddingSize;

    std::size_t totalProcessed =
        processedTokens + generatedTokens;

    if (totalProcessed % tokensPerBlock == 0) {
        std::size_t slot = gpu.allocateBlock(*this);
        memoryBlocks.push_back(slot);
        reservedBlocks--;
    }

    if (processedTokens < tokens.size()) {
        processedTokens++;
    } else {
        generatedTokens++;
    }

    std::vector<std::string> prefix(
        tokens.begin(),
        tokens.begin() + processedTokens
    );

    for (std::size_t i = 0; i < generatedTokens; i++) {
        prefix.push_back(outputTokens[i]);
    }

    std::size_t slot = memoryBlocks.back();

    std::size_t offset =
        (totalProcessed % tokensPerBlock) * embeddingSize;

    model.generateTokenEmbedding(prefix, gpu, slot, offset);

    if (prefix.size() % tokensPerBlock == 0) {
        gpu.assignPrefixToBlock(
            slot, prefix, model.getModelId()
        );
    }
}

void GenerativeRequest::allocateOneBlock(
    GPU& gpu, std::size_t slot) const {

    gpu.addMemoryBlock(*this, slot);
}

std::string GenerativeRequest::getOutput() const {
    std::string result = "";

    for (std::size_t i = 0; i < generatedTokens; i++) {
        if (i > 0) {
            result += " ";
        }

        result += outputTokens[i];
    }

    return result;
}

Request* GenerativeRequest::clone(const Model& model) const {

    std::unique_ptr<GenerativeRequest> copy(
        new GenerativeRequest(
            id,
            model,
            admissionCycle,
            "",
            scriptedOutput,
            maxOutputTokens
        )
    );

    copy->tokens = tokens;
    copy->memoryBlocks = memoryBlocks;
    copy->reservedBlocks = reservedBlocks;
    copy->outputTokens = outputTokens;
    copy->processedTokens = processedTokens;
    copy->generatedTokens = generatedTokens;

    return copy.release();
}

EmbeddingRequest::EmbeddingRequest(
    int id,
    const Model& model,
    int admissionCycle,
    std::string prompt)

    : Request(id, model, admissionCycle, prompt),
      processedTokens(0),
      output() {
}

bool EmbeddingRequest::finished() const {
    return processedTokens == tokens.size();
}

std::size_t EmbeddingRequest::getMaxNeededMemory() const {
    return 2 * model.getEmbeddingSize();
}

std::size_t EmbeddingRequest::getNeededMemory(
    const GPU& gpu) const {

    return getMaxNeededMemory();
}

std::vector<std::string>
EmbeddingRequest::getCacheableTokens() const {
    return {};
}

void EmbeddingRequest::allocateOneBlock(
    GPU& gpu, std::size_t slot) const {

    gpu.addMemoryBlock(*this, slot);
}

void EmbeddingRequest::reserveMemory(GPU& gpu) {
    reservedBlocks = gpu.getBlocksForMemory(
        getNeededMemory(gpu)
    );

    gpu.reserve(reservedBlocks, *this);

    while (reservedBlocks > 0) {
        std::size_t slot = gpu.allocateBlock(*this);
        memoryBlocks.push_back(slot);
        reservedBlocks--;
    }
}

std::size_t EmbeddingRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}

void EmbeddingRequest::act(GPU& gpu) {
    if (finished()) {
        return;
    }

    std::size_t embeddingSize = model.getEmbeddingSize();
    std::size_t blockSize = gpu.getBlockSize();

    std::size_t sumSlot = memoryBlocks[0];

    std::size_t tokenSlot =
        memoryBlocks[embeddingSize / blockSize];

    std::size_t tokenOffset =
        embeddingSize % blockSize;

    std::vector<std::string> currentToken;
    currentToken.push_back(tokens[processedTokens]);

    model.generateTokenEmbedding(
        currentToken, gpu, tokenSlot, tokenOffset
    );

    for (std::size_t i = 0; i < embeddingSize; i++) {
        int sum = gpu.readFromBlock(sumSlot, i);

        int value = gpu.readFromBlock(
            tokenSlot, tokenOffset + i
        );

        gpu.writeToBlock(sumSlot, i, sum + value);
    }

    processedTokens++;

    if (finished()) {
        for (std::size_t i = 0; i < embeddingSize; i++) {
            int sum = gpu.readFromBlock(sumSlot, i);

            output.push_back(
                sum / static_cast<int>(tokens.size())
            );
        }
    }
}

std::string EmbeddingRequest::getOutput() const {
    std::string result = "[";

    for (std::size_t i = 0; i < output.size(); i++) {
        if (i > 0) {
            result += ", ";
        }

        result += std::to_string(output[i]);
    }

    result += "]";

    return result;
}

Request* EmbeddingRequest::clone(const Model& model) const {

    std::unique_ptr<EmbeddingRequest> copy(
        new EmbeddingRequest(
            id,
            model,
            admissionCycle,
            ""
        )
    );

    copy->tokens = tokens;
    copy->memoryBlocks = memoryBlocks;
    copy->reservedBlocks = reservedBlocks;
    copy->processedTokens = processedTokens;
    copy->output = output;

    return copy.release();
}