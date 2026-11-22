
#include "Server.h"
#include <iostream>
#include <sstream>

int main() {
    Server original("test_multi_gpu.json");

    // Run two cycles before copying
    original.step();
    original.step();

    // Copy the server in the middle of execution
    Server copied(original);

    // Capture the original server's output
    std::ostringstream originalOutput;
    std::ostringstream copiedOutput;

    std::streambuf* oldBuffer = std::cout.rdbuf();

    std::cout.rdbuf(originalOutput.rdbuf());
    original.serve();

    // Capture the copied server's output
    std::cout.rdbuf(copiedOutput.rdbuf());
    copied.serve();

    // Restore standard output
    std::cout.rdbuf(oldBuffer);

    std::cout << "Original server:\n";
    std::cout << originalOutput.str();

    std::cout << "\nCopied server:\n";
    std::cout << copiedOutput.str();

    if (originalOutput.str() == copiedOutput.str()) {
        std::cout << "\nCOPY TEST PASSED!\n";
    } else {
        std::cout << "\nCOPY TEST FAILED!\n";
        return 1;
    }

    return 0;
}
