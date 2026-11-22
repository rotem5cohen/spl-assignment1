#include "SchedulingPolicy.h"
#include "Request.h"
#include "GPU.h"
#include "Model.h"
#include <algorithm>

std::vector<int> ArrivalOrderPolicy::choose(
    const std::vector<const Request*>& eligible,
    const std::vector<GPU>& gpus
) const {

    std::vector<int> indices;

    for (std::size_t i = 0; i < eligible.size(); i++) {
        indices.push_back(static_cast<int>(i));
    }

    std::sort(indices.begin(), indices.end(),
        [&eligible](int a, int b) {

            if (eligible[a]->getAdmissionCycle() !=
                eligible[b]->getAdmissionCycle()) {

                return eligible[a]->getAdmissionCycle() <
                       eligible[b]->getAdmissionCycle();
            }

            return eligible[a]->getId() <
                   eligible[b]->getId();
        }
    );

    return indices;
}

SchedulingPolicy* ArrivalOrderPolicy::clone() const {
    return new ArrivalOrderPolicy(*this);
}

std::vector<int> ShortestJobPolicy::choose(
    const std::vector<const Request*>& eligible,
    const std::vector<GPU>& gpus
) const {

    std::vector<int> indices;

    for (std::size_t i = 0; i < eligible.size(); i++) {
        indices.push_back(static_cast<int>(i));
    }

    std::sort(indices.begin(), indices.end(),
        [&eligible, &gpus](int a, int b) {

            std::size_t blocksA = gpus[0].getBlocksForMemory(
                eligible[a]->getMaxNeededMemory()
            );

            std::size_t blocksB = gpus[0].getBlocksForMemory(
                eligible[b]->getMaxNeededMemory()
            );

            if (blocksA != blocksB) {
                return blocksA < blocksB;
            }

            return eligible[a]->getId() <
                   eligible[b]->getId();
        }
    );

    return indices;
}


SchedulingPolicy* ShortestJobPolicy::clone() const {
    return new ShortestJobPolicy(*this);
}

static std::size_t getLongestPrefix(
    const Request& request,
    const std::vector<GPU>& gpus) {

    std::size_t longest = 0;

    for (const GPU& gpu : gpus) {

        PrefixMatch match = gpu.findLongestCachedPrefix(
            request.getModel().getModelId(),
            request.getModel().getEmbeddingSize(),
            request.getCacheableTokens()
        );

        if (match.matchedTokens > longest) {
            longest = match.matchedTokens;
        }
    }

    return longest;
}

std::vector<int> LongestPrefixPolicy::choose(
    const std::vector<const Request*>& eligible,
    const std::vector<GPU>& gpus
) const {

    std::vector<int> indices;
    std::vector<std::size_t> prefixes;

    for (std::size_t i = 0; i < eligible.size(); i++) {
        indices.push_back(static_cast<int>(i));

        prefixes.push_back(
            getLongestPrefix(*eligible[i], gpus)
        );
    }

    std::sort(indices.begin(), indices.end(),
        [&eligible, &prefixes](int a, int b) {

            if (prefixes[a] != prefixes[b]) {
                return prefixes[a] > prefixes[b];
            }

            return eligible[a]->getId() <
                   eligible[b]->getId();
        }
    );

    return indices;
}

SchedulingPolicy* LongestPrefixPolicy::clone() const {
    return new LongestPrefixPolicy(*this);
}
