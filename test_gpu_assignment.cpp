
#include "GPU.h"
#include "Model.h"
#include "Request.h"
#include <iostream>

int main() {
    Model model("llama", 4);

    GPU original(3, 8);

    GenerativeRequest request(
        1, model, 0, "alpha beta", "", 0
    );

    // Admit the request and fill one KV block
    request.reserveMemory(original);
    request.act(original);
    request.act(original);

    // Find the cached block
    PrefixMatch match = original.findLongestCachedPrefix(
        model.getModelId(),
        model.getEmbeddingSize(),
        request.getCacheableTokens()
    );

    if (match.blockIds.size() != 1) {
        std::cout << "SETUP FAILED\n";
        return 1;
    }

    std::size_t slot = match.blockIds[0];

    // Copy assignment into an existing GPU
    GPU copied(2, 8);
    copied = original;

    bool copyPassed =
        copied.getBlocksNum() == original.getBlocksNum() &&
        copied.getBlockRefCount(slot) == 1 &&
        copied.findLongestCachedPrefix(
            model.getModelId(),
            model.getEmbeddingSize(),
            request.getCacheableTokens()
        ).matchedTokens == 2;

    // Releasing the copy must not affect the original
    copied.release(slot);

    bool independent =
        copied.getBlockRefCount(slot) == 0 &&
        original.getBlockRefCount(slot) == 1;

    // Self-assignment
    original = original;

    bool selfAssignmentPassed =
        original.getBlockRefCount(slot) == 1;

    std::cout << "GPU Copy Assignment: "
              << (copyPassed ? "PASSED" : "FAILED") << "\n";

    std::cout << "Independent Memory: "
              << (independent ? "PASSED" : "FAILED") << "\n";

    std::cout << "Self Assignment: "
              << (selfAssignmentPassed ? "PASSED" : "FAILED") << "\n";

    // Release original request's references
    request.deallocateMemory(original);

    if (!copyPassed || !independent || !selfAssignmentPassed) {
        return 1;
    }

    std::cout << "\nGPU ASSIGNMENT TEST PASSED!\n";
    return 0;
}
