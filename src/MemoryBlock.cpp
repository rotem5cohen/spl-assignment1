#include "MemoryBlock.h"

MemoryBlock::MemoryBlock(size_t size)
    : size(size),
      refCount(0),
      data(new int[size]()) {
}

MemoryBlock::~MemoryBlock() {
    delete[] data;
}

MemoryBlock::MemoryBlock(const MemoryBlock& other)
    : size(other.size),
      refCount(0),
      data(new int[other.size]()) {

    for (size_t i = 0; i < size; i++) {
        data[i] = other.data[i];
    }
}

MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other) {
    if (this == &other) {
        return *this;
    }

    int* newData = new int[other.size]();

    for (size_t i = 0; i < other.size; i++) {
        newData[i] = other.data[i];
    }

    delete[] data;

    data = newData;
    size = other.size;
    refCount = 0;

    return *this;
}

MemoryBlock::MemoryBlock(MemoryBlock&& other) noexcept
    : size(other.size),
      refCount(other.refCount),
      data(other.data) {

    other.size = 0;
    other.refCount = 0;
    other.data = nullptr;
}

MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    delete[] data;

    size = other.size;
    refCount = other.refCount;
    data = other.data;

    other.size = 0;
    other.refCount = 0;
    other.data = nullptr;

    return *this;
}

void MemoryBlock::writeToBlock(size_t offset, int value) {
    data[offset] = value;
}

int MemoryBlock::readFromBlock(size_t offset) const {
    return data[offset];
}

void MemoryBlock::acquire() {
    refCount++;
}

IdleAction MemoryBlock::release() {
    refCount--;

    if (refCount == 0) {
        return onLastRelease();
    }

    return IdleAction::KeepCached;
}

int MemoryBlock::getRefCount() const {
    return refCount;
}

KVBlock::KVBlock(size_t size)
    : MemoryBlock(size) {
}

IdleAction KVBlock::onLastRelease() const {
    return IdleAction::KeepCached;
}

MemoryBlock* KVBlock::clone() const {
    return new KVBlock(*this);
}

EmbeddingWorkspace::EmbeddingWorkspace(size_t size)
    : MemoryBlock(size) {
}

IdleAction EmbeddingWorkspace::onLastRelease() const {
    return IdleAction::Destroy;
}

MemoryBlock* EmbeddingWorkspace::clone() const {
    return new EmbeddingWorkspace(*this);
}



