
#include "Server.h"
#include <iostream>
#include <exception>

int main() {
    try {
        Server server("test_invalid.json");

        std::cout << "TEST FAILED: No exception\n";
        return 1;

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << "\n";
        std::cout << "EXCEPTION TEST PASSED!\n";
    }

    return 0;
}
