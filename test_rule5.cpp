
#include "Server.h"
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

// Run the server and capture its output
std::string runServer(Server& server) {
    std::ostringstream output;

    std::streambuf* originalBuffer = std::cout.rdbuf();
    std::cout.rdbuf(output.rdbuf());

    try {
        server.serve();
    } catch (...) {
        std::cout.rdbuf(originalBuffer);
        throw;
    }

    std::cout.rdbuf(originalBuffer);
    return output.str();
}

int main() {

    Server original("test_multi_gpu.json");

    // Advance the original server by two cycles
    original.step();
    original.step();

    // Test 1: Copy Assignment
    Server copyAssigned("test_cache.json");
    copyAssigned = original;

    // Test 2: Move Constructor
    Server moveSource(original);
    Server moveConstructed(std::move(moveSource));

    // Test 3: Move Assignment
    Server moveAssignSource(original);
    Server moveAssigned("test_cache.json");
    moveAssigned = std::move(moveAssignSource);

    // Continue all servers independently
    std::string expected = runServer(original);

    std::string copyResult = runServer(copyAssigned);
    std::string moveCtorResult = runServer(moveConstructed);
    std::string moveAssignResult = runServer(moveAssigned);

    bool copyPassed = (copyResult == expected);
    bool moveCtorPassed = (moveCtorResult == expected);
    bool moveAssignPassed = (moveAssignResult == expected);

    // Moved-from servers must be empty
    std::string emptySummary =
        "summary: cycles=0 cache_hits=0 evictions=0\n";

    bool movedFromPassed =
        runServer(moveSource) == emptySummary &&
        runServer(moveAssignSource) == emptySummary;

    std::cout << "Copy Assignment: "
              << (copyPassed ? "PASSED" : "FAILED") << "\n";

    std::cout << "Move Constructor: "
              << (moveCtorPassed ? "PASSED" : "FAILED") << "\n";

    std::cout << "Move Assignment: "
              << (moveAssignPassed ? "PASSED" : "FAILED") << "\n";

    std::cout << "Moved-from Servers: "
              << (movedFromPassed ? "PASSED" : "FAILED") << "\n";

    if (!copyPassed || !moveCtorPassed ||
        !moveAssignPassed || !movedFromPassed) {

        std::cout << "\nRULE OF 5 TEST FAILED!\n";
        return 1;
    }

    std::cout << "\nALL RULE OF 5 TESTS PASSED!\n";
    return 0;
}
