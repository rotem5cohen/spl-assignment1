#include "MemoryBlock.h"
#include <iostream>
#include <utility>

int main() {
    KVBlock original(4);
    original.writeToBlock(0, 42);
    original.acquire();
    original.acquire();

    KVBlock moved(std::move(original));

    bool constructorPassed =
        moved.readFromBlock(0) == 42 &&
        moved.getRefCount() == 2 &&
        original.getRefCount() == 0;

    KVBlock target(4);
    target = std::move(moved);

    bool assignmentPassed =
        target.readFromBlock(0) == 42 &&
        target.getRefCount() == 2 &&
        moved.getRefCount() == 0;

    std::cout << "Move Constructor: "
              << (constructorPassed ? "PASSED" : "FAILED")
              << std::endl;

    std::cout << "Move Assignment: "
              << (assignmentPassed ? "PASSED" : "FAILED")
              << std::endl;

    return (constructorPassed && assignmentPassed) ? 0 : 1;
}