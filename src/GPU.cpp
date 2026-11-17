#include "GPU.h"
#include "MemoryBlock.h"
#include "Request.h"
#include <stdexcept>

GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum),
      blockSize(blockSize),
      memory(blocksNum, nullptr),
      totalReserved(0),
      reservations(),
      cacheHits(0),
      evictions(0),
      modelIds(blocksNum),
      prefixKeys(blocksNum) {
}

std::size_t GPU::availableSlots() const {
    std::size_t activeBlocks = 0;

    for (std::size_t i = 0; i < blocksNum; i++) {
        if (memory[i] != nullptr &&
            memory[i]->getRefCount() > 0) {

            activeBlocks++;
        }
    }

    return blocksNum - activeBlocks - totalReserved;
}

std::size_t GPU::getBlocksForMemory(std::size_t memory) const {
    return memory / blockSize + (memory % blockSize != 0);
}

void GPU::reserve(std::size_t n, const Request& request) {
    if (n == 0) {
        return;
    }

    reservations[request.getId()] += n;
    totalReserved += n;
}

void GPU::unreserve(std::size_t n, const Request& request) {
    if (n == 0) {
        return;
    }

    reservations.at(request.getId()) -= n;
    totalReserved -= n;

    if (reservations.at(request.getId()) == 0) {
        reservations.erase(request.getId());
    }
}

void GPU::addMemoryBlock(const GenerativeRequest& request,
                         std::size_t slot) {
    memory[slot] = new KVBlock(blockSize);
}

void GPU::addMemoryBlock(const EmbeddingRequest& request,
                         std::size_t slot) {
    memory[slot] = new EmbeddingWorkspace(blockSize);
}

std::size_t GPU::allocateBlock(const Request& request) {
    if (reservations.count(request.getId()) == 0 ||
        reservations.at(request.getId()) == 0) {
        throw std::logic_error("No reserved blocks");
    }

    std::size_t slot = blocksNum;

    for (std::size_t i = 0; i < blocksNum; i++) {
        if (memory[i] == nullptr) {
            slot = i;
            break;
        }
    }

    if (slot == blocksNum) {
        for (std::size_t i = 0; i < blocksNum; i++) {
            if (memory[i]->getRefCount() == 0) {
                slot = i;

                delete memory[i];
                memory[i] = nullptr;

                modelIds[i].clear();
                prefixKeys[i].clear();

                evictions++;
                break;
            }
        }
    }

    if (slot == blocksNum) {
        throw std::logic_error("No available slot");
    }

    request.allocateOneBlock(*this, slot);

    memory[slot]->acquire();
    unreserve(1, request);

    return slot;
}

int GPU::readFromBlock(std::size_t blockId,
                       std::size_t offset) const {
    return memory[blockId]->readFromBlock(offset);
}

void GPU::writeToBlock(std::size_t blockId,
                       std::size_t offset, int value) {
    memory[blockId]->writeToBlock(offset, value);
}

void GPU::acquire(std::size_t slot) {
    memory[slot]->acquire();
}

void GPU::release(std::size_t slot) {
    IdleAction action = memory[slot]->release();

    if (memory[slot]->getRefCount() > 0) {
        return;
    }

    if (action == IdleAction::Destroy ||
        prefixKeys[slot].empty()) {

        delete memory[slot];
        memory[slot] = nullptr;

        modelIds[slot].clear();
        prefixKeys[slot].clear();
    }
}

void GPU::assignPrefixToBlock(
    std::size_t block,
    const std::vector<std::string>& prefixTokens,
    const std::string& modelId) {

    for (std::size_t i = 0; i < blocksNum; i++) {
        if (memory[i] != nullptr &&
            modelIds[i] == modelId &&
            prefixKeys[i] == prefixTokens) {

            return;
        }
    }

    modelIds[block] = modelId;
    prefixKeys[block] = prefixTokens;
}

PrefixMatch GPU::findLongestCachedPrefix(
    const std::string& modelId,
    std::size_t embeddingSize,
    const std::vector<std::string>& promptTokens) const {

    PrefixMatch result{0, {}};

    std::size_t tokensPerBlock = blockSize / embeddingSize;

    for (std::size_t len = tokensPerBlock;
         len <= promptTokens.size();
         len += tokensPerBlock) {

        std::vector<std::string> currentPrefix(
            promptTokens.begin(),
            promptTokens.begin() + len);

        bool found = false;

        for (std::size_t i = 0; i < blocksNum; i++) {
            if (memory[i] != nullptr &&
                modelIds[i] == modelId &&
                prefixKeys[i] == currentPrefix) {

                result.blockIds.push_back(i);
                result.matchedTokens += tokensPerBlock;

                found = true;
                break;
            }
        }

        if (!found) {
            break;
        }
    }

    return result;
}

void GPU::copyMemoryFromOtherGPU(
    const GPU& other,
    const std::vector<std::size_t>& blocks) {

    std::vector<bool> protectedSlots(blocksNum, false);

  
    for (std::size_t sourceSlot : blocks) {
        for (std::size_t i = 0; i < blocksNum; i++) {
            if (memory[i] != nullptr &&
                modelIds[i] == other.modelIds[sourceSlot] &&
                prefixKeys[i] == other.prefixKeys[sourceSlot]) {

                protectedSlots[i] = true;
                break;
            }
        }
    }

  
    for (std::size_t sourceSlot : blocks) {

        bool exists = false;

        for (std::size_t i = 0; i < blocksNum; i++) {
            if (memory[i] != nullptr &&
                modelIds[i] == other.modelIds[sourceSlot] &&
                prefixKeys[i] == other.prefixKeys[sourceSlot]) {

                exists = true;
                break;
            }
        }

        if (exists) {
            continue;
        }

        std::size_t slot = blocksNum;

        for (std::size_t i = 0; i < blocksNum; i++) {
            if (memory[i] == nullptr) {
                slot = i;
                break;
            }
        }

        if (slot == blocksNum) {
            for (std::size_t i = 0; i < blocksNum; i++) {
                if (memory[i]->getRefCount() == 0 &&
                    !protectedSlots[i]) {

                    slot = i;

                    delete memory[i];
                    memory[i] = nullptr;

                    modelIds[i].clear();
                    prefixKeys[i].clear();

                    evictions++;
                    break;
                }
            }
        }

        if (slot == blocksNum) {
            throw std::logic_error("No available slot");
        }

        memory[slot] = other.memory[sourceSlot]->clone();

        modelIds[slot] = other.modelIds[sourceSlot];
        prefixKeys[slot] = other.prefixKeys[sourceSlot];

        protectedSlots[slot] = true;
    }
}