
#include "Server.h"
#include <iostream>
#include <sstream>
#include <string>

std::string captureOutput(Server& server) {
    std::ostringstream output;

    std::streambuf* oldBuffer = std::cout.rdbuf();
    std::cout.rdbuf(output.rdbuf());

    try {
        server.serve();
    } catch (...) {
        std::cout.rdbuf(oldBuffer);
        throw;
    }

    std::cout.rdbuf(oldBuffer);
    return output.str();
}

int main() {

    Server original("test_shared_blocks.json");

    // Execute cycles 0, 1 and 2
    original.step();
    original.step();
    original.step();

    // Both requests now share one active KV block
    Server copied(original);

    // Continue both servers independently
    std::string originalOutput = captureOutput(original);
    std::string copiedOutput = captureOutput(copied);

    std::cout << "Original server:\n";
    std::cout << originalOutput;

    std::cout << "\nCopied server:\n";
    std::cout << copiedOutput;

    if (originalOutput == copiedOutput) {
        std::cout << "\nSHARED COPY TEST PASSED!\n";
        return 0;
    }

    std::cout << "\nSHARED COPY TEST FAILED!\n";
    return 1;
}
